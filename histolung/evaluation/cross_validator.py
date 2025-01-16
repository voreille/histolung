from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


class CrossValidator:

    def __init__(self, dataset, evaluator, n_splits=5):
        self.dataset = dataset
        self.evaluator = evaluator
        self.n_splits = n_splits

    def cross_validate(self, model):
        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(self.dataset.metadata_df,
                                            self.dataset.metadata_df['label']):
            train_data = self.dataset.metadata_df.iloc[train_idx]
            val_data = self.dataset.metadata_df.iloc[val_idx]

            train_loader = DataLoader(self.dataset(train_data))
            val_loader = DataLoader(self.dataset(val_data))

            self.evaluator.test_loader = val_loader
            score = self.evaluator.evaluate(model)
            scores.append(score)

        return scores
