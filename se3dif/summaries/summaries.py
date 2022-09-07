from .sdf_summary import sdf_summary
from .denoising_summary import denoising_summary


class SummaryDict():
    def __init__(self, summaries):
        self.fields = summaries.keys()
        self.summaries = summaries

    def compute_summary(self, model, model_input, ground_truth, info , writer, iter, prefix=""):
        for field in self.fields:
            prefix_in = prefix + field
            self.summaries[field](model, model_input, ground_truth, info, writer, iter, prefix_in)


def get_summary(args, activate_summary=False):
    if activate_summary:
        summaries = {'sdf': sdf_summary, 'denoising': denoising_summary}
    else:
        summaries = {}
    summary_dict = SummaryDict(summaries=summaries)
    return summary_dict.compute_summary