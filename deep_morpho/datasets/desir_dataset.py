from enum import Enum
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms


from general.nn.dataloaders import dataloader_resolution_from_df
from general.utils import center_and_crop
from .datamodule_base import DataModule


DEFAULT_DATA_INFO = pd.read_csv('data/deep_morpho/desir/info.csv')
DEFAULT_DATA_SPONDI_INFO = pd.read_csv('data/deep_morpho/desir_from_spondidetect/info.csv')  # from spondidetect repo
DEFAULT_PREPROCESSING = None


class CropMethod(Enum):
    NONE = 0
    SEGM = 1
    ROI = 2



class DesirDatasetBase(DataModule, Dataset):

    def __init__(
            self,
            data_info=DEFAULT_DATA_INFO,
            label_col="bme_positive_r1",
            preprocessing_indep=DEFAULT_PREPROCESSING,
            preprocessing_both=None,
            center_and_crop_method: CropMethod = CropMethod.NONE,
            cropped_size: tuple = (256, 256),
        ):
        self.data_info = data_info[~data_info[label_col].isna()]
        self.preprocessing_indep = preprocessing_indep
        self.preprocessing_both = preprocessing_both
        self.label_col = label_col
        self.center_and_crop_method = center_and_crop_method
        self.cropped_size = cropped_size

    def __getitem__(self, idx: int):
        row = self.data_info.iloc[idx]

        ars = {}

        ars["segm"] = np.load(row.segm_path)
        ars["roi"] = np.load(row.roi_path)
        ars["t1_npy"] = np.load(row.t1_npy_path)
        ars["stir_npy"] = np.load(row.stir_npy_path)

        # segm = np.load(row.segm_path)
        # roi = np.load(row.roi_path)
        # t1_npy = np.load(row.t1_npy_path)
        # stir_npy = np.load(row.stir_npy_path)
        label = row[self.label_col] > 0  # bme is between 0 and 6 (4 quarters + intensity and depth). Only take the presence of BME in one quarter.

        # assert roi.sum() > 0
        assert ars["roi"].sum() > 0

        for v in ars.values():
            assert v.shape == ars["roi"].shape

        if row.side == "right":
            for k, v in ars.items():
                ars[k] = v[:, :v.shape[1]//2:-1] + 0
        else:
            for k, v in ars.items():
                ars[k] = v[:, :v.shape[1]//2:-1] + 0


        if self.center_and_crop_method.value == CropMethod.SEGM.value:
            mask = ars["segm"] + 0
            for k, v in ars.items():
                ars[k] = center_and_crop(v, mask, self.cropped_size)
        elif self.center_and_crop_method.value == CropMethod.ROI.value:
            mask = ars["roi"] + 0
            for k, v in ars.items():
                ars[k] = center_and_crop(v, mask, self.cropped_size)

        # assert segm.shape == roi.shape
        # assert roi.shape == segm.shape
        # assert t1_npy.shape == segm.shape
        # assert stir_npy.shape == segm.shape

        # if row.side == "right":
        #     roi = roi[:, :segm.shape[1]//2:-1] + 0
        #     t1_npy = t1_npy[:, :segm.shape[1]//2:-1] + 0
        #     stir_npy = stir_npy[:, :segm.shape[1]//2:-1] + 0
        #     segm = segm[:, :segm.shape[1]//2:-1] + 0
        # else:
        #     roi = roi[:, :segm.shape[1]//2:-1] + 0
        #     t1_npy = t1_npy[:, :segm.shape[1]//2:-1] + 0
        #     stir_npy = stir_npy[:, :segm.shape[1]//2:-1] + 0
        #     segm = segm[:, :segm.shape[1]//2:-1] + 0

        # if self.center_and_crop_method == CropMethod.SEGM:
        #     t1_npy,

        # big_array = np.stack([t1_npy, stir_npy, segm, roi,], axis=-1)
        big_array = np.stack([ars["t1_npy"], ars['stir_npy'], ars["segm"], ars["roi"],], axis=-1)
        big_tensor = transforms.ToTensor()(big_array).float()
        if self.preprocessing_both != None:
            big_tensor = self.preprocessing_both(big_tensor)

        t1_tensor = big_tensor[0]
        stir_tensor = big_tensor[1]

        if self.preprocessing_indep != None:
            t1_tensor = self.preprocessing_indep(t1_tensor)
            stir_tensor = self.preprocessing_indep(stir_tensor)

        tensor_array = torch.tensor(np.stack([t1_tensor, stir_tensor]))

        segm_prev = big_tensor[2].unsqueeze(0)
        segm = segm_prev + 0
        values = segm.unique(sorted=True)
        for value_idx, value in enumerate(values):
            segm[segm_prev == value] = value_idx

        roi = big_tensor[3].unsqueeze(0)
        label = torch.tensor(label).float()

        return (tensor_array, segm, roi), label

    def __len__(self):
        return len(self.data_info)

    @classmethod
    def get_loader(cls, batch_size: int, data_info: pd.DataFrame = DEFAULT_DATA_INFO,  num_workers: int = 0, shuffle: bool = True, **kwargs) -> DataLoader:
        return dataloader_resolution_from_df(
            df=data_info,
            dataset=cls,
            dataset_args=kwargs,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase", ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        args = experiment.args
        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(
            experiment, [
                "batch_size", "num_workers", "shuffle", "data_info",
            ]
        )

        df_all = DEFAULT_DATA_INFO
        df_train = df_all[np.isin(df_all, args["train_patients"])]
        df_val = df_all[np.isin(df_all, args["val_patients"])]
        df_test = df_all[np.isin(df_all, args["test_patients"])]

        experiment.log_console(f"Train: {len(df_train['patient_id'].unique())} patients, {len(df_train)} half-slices")
        experiment.log_console(f"Val: {len(df_val['patient_id'].unique())} patients, {len(df_val)} half-slices")
        experiment.log_console(f"Test: {len(df_test['patient_id'].unique())} patients, {len(df_test)} half-slices")
        # experiment.log_console(f"Preprocessing: {args['preprocessing']}")

        trainloader = cls.get_loader(data_info=df_train, shuffle=True, batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs)
        valloader = cls.get_loader(data_info=df_val, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs)
        testloader = cls.get_loader(data_info=df_test, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs)

        return trainloader, valloader, testloader


class DesirDatasetHalfSlice(DesirDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["center_and_crop_method"] = CropMethod.NONE
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        return tensor_array, label


class DesirDatasetHalfSliceAndSegm(DesirDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["center_and_crop_method"] = CropMethod.SEGM
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        return (tensor_array, torch.cat([segm == 1, segm == 2], axis=0).float()), label

class DesirDatasetMerged(DesirDatasetBase):
    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        return tensor_array * roi, label


class DesirDatasetMergedSegm(DesirDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["center_and_crop_method"] = CropMethod.SEGM
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        return tensor_array * (segm != 0).float(), label

class DesirDatasetMergedSegmChannel(DesirDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["center_and_crop_method"] = CropMethod.SEGM
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        return torch.cat([tensor_array, (segm != 0).float()]), label

class DesirDatasetRoiChannel(DesirDatasetBase):
    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        return torch.cat([tensor_array, roi]), label

class DesirDatasetSegmChannel(DesirDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["center_and_crop_method"] = CropMethod.SEGM
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        (tensor_array, segm, roi), label = super().__getitem__(idx)
        segm_1 = segm == 1
        segm_2 = segm == 2

        return torch.cat([tensor_array, segm_1, segm_2]), label

class DesirFromSpondidetectDataset(DataModule, Dataset):
    def __init__(
        self,
        data_info: pd.DataFrame = DEFAULT_DATA_SPONDI_INFO,
        preprocessing_both=None,
    ):
        self.data_info = data_info
        self.preprocessing_both = preprocessing_both

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int):
        row = self.data_info.iloc[idx]
        tensor = torch.load(row.input_path).cpu()
        if self.preprocessing_both != None:
            tensor = self.preprocessing_both(tensor)

        label = torch.tensor(row.label).float()
        return tensor, label

    @classmethod
    def get_loader(
        cls, batch_size: int, num_workers: int = 0, shuffle: bool = True, **kwargs
    ) -> DataLoader:
        return DataLoader(cls(**kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,)

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase", ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        args = experiment.args
        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(
            experiment, [
                "batch_size", "num_workers", "shuffle", "data_info",
            ]
        )

        df_all = DEFAULT_DATA_SPONDI_INFO
        df_train = df_all[np.isin(df_all, args["train_patients"])]
        df_val = df_all[np.isin(df_all, args["val_patients"])]
        df_test = df_all[np.isin(df_all, args["test_patients"])]

        experiment.log_console(f"Train: {len(df_train['patient_id'].unique())} patients, {len(df_train)} half-slices")
        experiment.log_console(f"Val: {len(df_val['patient_id'].unique())} patients, {len(df_val)} half-slices")
        experiment.log_console(f"Test: {len(df_test['patient_id'].unique())} patients, {len(df_test)} half-slices")
        experiment.log_console(f"Preprocessing: {args['preprocessing']}")

        trainloader = cls.get_loader(data_info=df_train,  shuffle=True, batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs)
        valloader = cls.get_loader(data_info=df_val,  shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs)
        testloader = cls.get_loader(data_info=df_test,  shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs)

        return trainloader, valloader, testloader
