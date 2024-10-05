from cleanfid import fid


score = fid.compute_fid('cifar_auto_np/', dataset_name="cifar10", dataset_res=32, dataset_split='train')
print(score)