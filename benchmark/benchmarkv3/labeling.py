import pandas as pd
import os
from random import choice

path = os.path.dirname(__file__)
problemset_path = os.path.join(path, 'df.csv')
qa_path = os.path.join(path, "q&a.csv")

# problemset_df = pd.read_csv(problemset_path)
# problemset_df = problemset_df.rename({"Unnamed: 0": "id"}, axis=1)
# print(problemset_df)
# problemset_df.to_csv(problemset_path, index=False)

# qa_df = pd.DataFrame(columns=["Anchor", "Golden", "Silver", "Wrong"]) 
# qa_df.to_csv(qa_path, index=False)

class LabelingTool:
    def __init__(self, path = path, problemset_path = problemset_path, qa_path = qa_path, n = 4):
        self.n = n
        self.path = path
        self.problemset_path = problemset_path
        self.qa_path = qa_path
        self.problemset_df = pd.read_csv(self.problemset_path)
        self.qa_df = pd.read_csv(self.qa_path)
        self.label_dfs = {"Anchor": None,
                          "Golden": None,
                          "Silver": None,
                          "Wrong": None}
        self.methods = {
            "Anchor": self.filter_golden,
            "Golden": self.filter_golden,
            "Silver": self.filter_silver,
            "Wrong": self.filter_wrong
        }

    def get_subtopics(self):
        """Returns list of topics from problemset dataframe"""
        return self.problemset_df['TopicMetadata'].unique()
    
    def filter_golden(self, subtopic):
        """
        Filters from dataset only subtopic
        subtopic: str - Full name of subtopic as defined in column "Topic Metadata". If none or empty filters by random subtopic.
        n: int - Number of problems to sample
        """
        filtered_df = self.problemset_df[self.problemset_df['used'] == False]
        if subtopic == "" or subtopic is None:
            subtopic = choice(self.get_subtopics())
        filtered_df = filtered_df[filtered_df['TopicMetadata'] == subtopic]
        filtered_df = filtered_df.sample(n=min(self.n, len(filtered_df)))

        return filtered_df
    
    def filter_silver(self, subtopic):
        """
        Filters df leaving only rows with topic of subtopic. Excludes instance of subtopic itself. Topic is the first entrance of TopicMetadata.
        subtopic: str - Full name of subtopic as defined in column "Topic Metadata".
        n: int - Number of problems to sample 
        """
        assert subtopic != "" and subtopic is not None
        topic = subtopic.split("->")[0].strip('(').strip(')')
        filtered_df = self.problemset_df[self.problemset_df['used'] == False]
        filtered_df = filtered_df[filtered_df['Topic'] == topic]
        filtered_df = filtered_df[filtered_df['TopicMetadata'] != subtopic]
        filtered_df = filtered_df.sample(n=min(self.n, len(filtered_df)))
        return filtered_df
    
    def filter_wrong(self, subtopic):
        """
        Filters df leaving only rows with topic different from topic of subtopic. Topic is the first entrance of TopicMetadata.
        subtopic: str - Full name of subtopic as defined in column "Topic Metadata".
        n: int - Number of problems to sample 
        """
        assert subtopic != "" and subtopic is not None
        topic = subtopic.split("->")[0].strip('(').strip(')')
        filtered_df = self.problemset_df[self.problemset_df['used'] == False]
        filtered_df = filtered_df[filtered_df['Topic'] != topic]
        filtered_df = filtered_df.sample(n=min(self.n, len(filtered_df)))

        return filtered_df
    
    def labeling_options_init(self, subtopic):
        """
        Returns all of the tables needed to make a q&a row.
        subtopic: str - Full name of subtopic as defined in column "Topic Metadata".
        n: int - Number of problems to sample 
        """
        anchor = self.filter_golden(subtopic)
        if subtopic == "" or subtopic is None:
            subtopic = anchor['TopicMetadata'].unique()[0]
        golden = self.filter_golden(subtopic)
        silver = self.filter_silver(subtopic)
        wrong = self.filter_wrong(subtopic)

        self.label_dfs = {"Golden": golden,
                          "Anchor": anchor,
                          "Silver": silver,
                          "Wrong": wrong}
    
    def resample(self, key):
        method = self.methods[key]
        og_sample = self.label_dfs[key]
        subtopic = og_sample['TopicMetadata'].unique()[0]
        resamlped_df = method(subtopic)
        self.label_dfs[key] = resamlped_df
    
    def add_qa(self, ids: dict):
        tmp_df = pd.DataFrame([ids])
        self.qa_df = pd.concat([self.qa_df, tmp_df], ignore_index=True)

        for id in ids.values():
            print(id)
            mask = self.problemset_df['id'] == id
            self.problemset_df.loc[mask, 'used'] = True
    
    def save(self):
        self.qa_df.to_csv(self.qa_path, index=False)
        self.problemset_df.to_csv(self.problemset_path, index=False)

# tool = LabelingTool()
# subtopic = "(Divisibility)->(Divisibility)"
# # print(tool.filter_golden(subtopic))
# tool.add_qa({"Golden": 1, "Anchor": 1, "Silver": 1, "Wrong": 1})
# print(tool.qa_df)
# print(tool.problemset_df)