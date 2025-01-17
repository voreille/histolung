from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from histolung.evaluation.datasets import LungHist700DatasetManager, TileDataset


class CrossValidator:

    def __init__(self, dataset_manager, evaluator, n_splits=5):
        self.dataset_manager = dataset_manager
        self.evaluator = evaluator
        self.n_splits = n_splits

    def compute_embeddings(self, model, tiles_loader):
        tile_ids = []
        for batch in tiles_loader:
            images, batch_tile_ids = batch
            embeddings = model(images).to('cpu').detach().numpy()
            tile_ids.append(batch_tile_ids)

        embeddings = np.concatenate(embeddings, axis=0)
        tiles_ids = tile_ids

        return embeddings, tile_ids

    def cross_validate(self, model):
        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True,
                              random_state=42)
        scores = []
        tiles_loader = DataLoader(TileDataset(self.dataset_manager.tile_paths))
        embeddings, tile_ids = self.compute_embeddings(model, tiles_loader)

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
