import pandas as pd
import glob

from data_filter.CCF_AIOps_challenge_2022.base.base_class import BaseClass


class GroundTruthDao(BaseClass):
    def __init__(self):
        super().__init__()

    def get_ground_truth(self, dataset_type: str):
        result_dict = dict()

        data_base_path = f'{self.config.data_dict["ground_truth"][dataset_type]}'

        file_list = glob.glob(f'{data_base_path}/*.csv')
        for file in file_list:
            if dataset_type == 'train_valid':
                date = '2022-' + file.split('/')[-1].replace('.csv', '').split('2022-')[-1]
                cloud_bed = 'cloudbed-' + file.split('/')[-1].replace('.csv', '').split('-2022')[0].split('-')[-1]
            else:
                date = '2022-' + file.split('/')[-1].replace('.csv', '').split('2022-')[-1]
                cloud_bed = 'cloudbed'
            if date not in result_dict.keys():
                result_dict[date] = dict()
            ground_truth_df = pd.read_csv(file)
            result_dict[date][cloud_bed] = ground_truth_df.to_dict('list')
        return result_dict


if __name__ == '__main__':
    ground_truth_dao = GroundTruthDao()
    ground_truth_dao.get_ground_truth('train_valid')
