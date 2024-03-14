import argparse
import math
import operator
import os
import re
from collections import defaultdict

import nltk
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 200)

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def clean_phrase(line):
    return re.sub(r"[^\w\s]", "", line.strip().lower())


def ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def measure1(v1, v2):
    return v1 - v2


def measure2(v1, v2):
    return max(v1 - v2, 0)


def cosine_similarity(v1, v2):
    """Compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"""
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def parse_file(filename, n):
    with open(filename, "r") as f:
        results = defaultdict(int)
        for line in f:
            words = clean_phrase(line).split(" ")
            for ngram in ngrams(words, n):
                phrase = " ".join(word.strip() for word in ngram if len(word) > 0)
                results[phrase] += 1
    return results


def load_skills(filename):
    with open(filename, "r") as f:
        skills = [clean_phrase(line) for line in f]
    return list(set(skills))  # remove duplicates


def build_ngram_distribution(filename):
    n_s = [1, 2, 3]  # mono-, bi-, and tri-grams
    dist = {}
    for n in n_s:
        dist.update(parse_file(filename, n))
    return dist


class Extractor:
    def __init__(self, job_description_file, cv_file):
        self.softskills = load_skills("softskills.txt")
        self.hardskills = load_skills("hardskills.txt")
        self.jb_distribution = build_ngram_distribution(job_description_file)
        self.cv_distribution = build_ngram_distribution(cv_file)
        self.base_path = os.path.dirname(job_description_file)
        self.results_filename = self.base_path + "/run_results.csv"
        self.table = pd.DataFrame()
        self.outFile = "Extracted_keywords.csv"

    def send_to_file(self):
        try:
            os.remove(self.outFile)
        except OSError:
            pass
        df = pd.DataFrame(
            self.table, columns=["type", "skill", "job", "cv", "m1", "m2"]
        )
        df_sorted = df.sort_values(by=["job", "cv"], ascending=[False, False])
        df_sorted.to_csv(self.outFile, index=False)

    def print_measures(self):
        print(f"Measure 1: {self.table["Difference"].sum()}")
        print(f"Measure 2: {self.table["Modified Frequency"].sum()}")

        v1 = self.table["Frequency in Job Description"]
        v2 = self.table["Frequency in CV"]
        print(f"Measure 3 (cosine sim): {cosine_similarity(v1.values, v2.values)}")

        for skill in ["hard", "soft", "general"]:
            v1 = self.table[self.table["Skill Type"] == skill][
                "Frequency in Job Description"
            ]
            v2 = self.table[self.table["Skill Type"] == skill]["Frequency in CV"]
            print(
                f"Cosine similarity for {skill} skills: {cosine_similarity(v1.values, v2.values)}"
            )

    def save_measures(self):
        scores = []
        skills = ["hard", "soft", "general"]
        for skill in skills:
            v1 = self.table[self.table["Skill Type"] == skill][
                "Frequency in Job Description"
            ]
            v2 = self.table[self.table["Skill Type"] == skill]["Frequency in CV"]
            scores.append(cosine_similarity(v1.values, v2.values))

        # Define the header
        header = ",".join(["Run number"] + skills)

        # Initialize run number
        run_number = 0

        # Check if the file exists
        if os.path.exists(self.results_filename):
            # Read the file to find the last run number
            with open(self.results_filename, "r") as file:
                lines = file.readlines()
                if len(lines) > 1:  # Check if there are any runs logged
                    last_line = lines[-1]
                    run_number = int(last_line.split(",")[0])

        # Increment the run number
        run_number += 1

        # Append the new data to the file
        with open(self.results_filename, "a") as file:
            # If the file was newly created, write the header first
            if run_number == 1:
                file.write(f"{header}\n")
            file.write(",".join([str(score) for score in [run_number] + scores]) + "\n")

    def plot_progress(self):
        scores = pd.read_csv(self.results_filename, index_col=0)

        scores.plot(subplots=True, marker="o", linestyle="-")
        plt.show()

    def make_table(self):
        # I am interested in verbs, nouns, adverbs, and adjectives
        parts_of_speech = [
            "CD",
            "JJ",
            "JJR",
            "JJS",
            "MD",
            "NN",
            "NNS",
            "NNP",
            "NNPS",
            "RB",
            "RBR",
            "RBS",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
        ]
        graylist = "you will are be can want is do may thats come".split()
        tmp_table = []
        # look if the skills are mentioned in the job description and then in your cv

        for skill in self.hardskills:
            if skill in self.jb_distribution:
                count_jb = self.jb_distribution[skill]
                if skill in self.cv_distribution:
                    count_cv = self.cv_distribution[skill]
                else:
                    count_cv = 0
                m1 = measure1(count_jb, count_cv)
                m2 = measure2(count_jb, count_cv)
                tmp_table.append(["hard", skill, count_jb, count_cv, m1, m2])

        for skill in self.softskills:
            if skill in self.jb_distribution:
                count_jb = self.jb_distribution[skill]
                if skill in self.cv_distribution:
                    count_cv = self.cv_distribution[skill]
                else:
                    count_cv = 0
                m1 = measure1(count_jb, count_cv)
                m2 = measure2(count_jb, count_cv)
                tmp_table.append(["soft", skill, count_jb, count_cv, m1, m2])

        # And now for the general language of the job description:
        # Sort the distribution by the words most used in the job description
        general_language = sorted(
            self.jb_distribution.items(), key=operator.itemgetter(1), reverse=True
        )
        for word_frequency in general_language:
            skill = word_frequency[0]
            if (
                skill in self.hardskills
                or skill in self.softskills
                or skill in graylist
                or len(skill) == 0
            ):
                continue
            count_jb = word_frequency[1]
            tokens = nltk.word_tokenize(skill)
            parts = nltk.pos_tag(tokens)
            if all([parts[i][1] in parts_of_speech for i in range(len(parts))]):
                if skill in self.cv_distribution:
                    count_cv = self.cv_distribution[skill]
                else:
                    count_cv = 0
                m1 = measure1(count_jb, count_cv)
                m2 = measure2(count_jb, count_cv)
                tmp_table.append(["general", skill, count_jb, count_cv, m1, m2])
        self.table = pd.DataFrame(
            tmp_table,
            columns=[
                "Skill Type",
                "Skill",
                "Frequency in Job Description",
                "Frequency in CV",
                "Difference",
                "Modified Frequency",
            ],
        )

    def print_missing_skills(self):
        for skill in "hard soft general".split():
            print(f"Top 5 missing {skill} skills by difference")
            print(
                self.table[self.table["Skill Type"] == skill]
                .sort_values(by="Difference", ascending=False)
                .head(5)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare a job description with your cv"
    )
    parser.add_argument("--job", required=True, help="The job description")
    parser.add_argument("--cv", required=True, help="Your CV")
    args = parser.parse_args()

    K = Extractor(job_description_file=args.job, cv_file=args.cv)
    K.make_table()
    K.send_to_file()
    K.save_measures()
    K.print_measures()
    K.print_missing_skills()
    K.plot_progress()
