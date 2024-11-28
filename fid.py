from cleanfid import fid


score = fid.compute_fid('samples_215/', dataset_name="cifar10", dataset_res=32, dataset_split='train')
print(score)