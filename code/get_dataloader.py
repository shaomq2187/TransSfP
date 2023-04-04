import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
from dataloader.dataloader import TransSfPDataset


def get_dataloader(root_dir,augs_train,batch_size,num_workers,pin_memory):

    ######## train dataset concat ########
    dataset_middle_round_cup_black_background_12_28 = TransSfPDataset(dolp_dir = root_dir + '/real-world/middle-round-cup/params/DoLP',
                                                                                      aolp_dir = root_dir + '/real-world/middle-round-cup/params/AoLP',
                                                                                      synthesis_normals_dir= root_dir + '/real-world/middle-round-cup/synthesis-normals',
                                                                                      mask_dir= root_dir + '/real-world/middle-round-cup/masks',
                                                                                      label_dir= root_dir + '/real-world/middle-round-cup/normals-png', transform=augs_train)

    dataset_middle_square_cup_black_background_12_28 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/middle-square-cup/params/DoLP',
        aolp_dir= root_dir + '/real-world/middle-square-cup/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/middle-square-cup/synthesis-normals',
                                                                                       mask_dir= root_dir + '/real-world/middle-square-cup/masks',
                                                                                       label_dir= root_dir + '/real-world/middle-square-cup/normals-png', transform=augs_train)

    dataset_middle_white_cup_black_background_12_28 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/middle-white-cup/params/DoLP',
        aolp_dir= root_dir + '/real-world/middle-white-cup/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/middle-white-cup/synthesis-normals',
                                                                                      mask_dir= root_dir + '/real-world/middle-white-cup/masks',
                                                                                      label_dir= root_dir + '/real-world/middle-white-cup/normals-png', transform=augs_train)



    dataset_tiny_white_cup_black_background_12_28 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/tiny-cup/params/DoLP',
        aolp_dir= root_dir + '/real-world/tiny-cup/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/tiny-cup/synthesis-normals',
                                                                                    mask_dir= root_dir + '/real-world/tiny-cup/masks',
                                                                                    label_dir= root_dir + '/real-world/tiny-cup/normals-png', transform=augs_train)

    dataset_tiny_white_cup_edges_black_background_12_28 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/tiny-cup-edges/params/DoLP',
        aolp_dir= root_dir + '/real-world/tiny-cup-edges/params/AoLP',
        synthesis_normals_dir=  root_dir + '/real-world/tiny-cup-edges/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/tiny-cup-edges/masks',
                                                                                          label_dir= root_dir + '/real-world/tiny-cup-edges/normals-png', transform=augs_train)
    dataset_bird_back_1_20 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/bird-back/params/DoLP',
        aolp_dir= root_dir + '/real-world/bird-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/bird-back/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/bird-back/masks',
                                                                                          label_dir= root_dir + '/real-world/bird-back/normals-png', transform=augs_train)
    dataset_bird_front_1_20 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/bird-front/params/DoLP',
        aolp_dir= root_dir + '/real-world/bird-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/bird-front/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/bird-front/masks',
                                                                                          label_dir= root_dir + '/real-world/bird-front/normals-png', transform=augs_train)
    dataset_cat_front_1_20 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/cat-front/params/DoLP',
        aolp_dir= root_dir + '/real-world/cat-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/cat-front/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/cat-front/masks',
                                                                                          label_dir= root_dir + '/real-world/cat-front/normals-png', transform=augs_train)
    dataset_cat_back_1_20 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/cat-back/params/DoLP',
        aolp_dir= root_dir + '/real-world/cat-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/cat-back/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/cat-back/masks',
                                                                                          label_dir= root_dir + '/real-world/cat-back/normals-png', transform=augs_train)
    dataset_hemi_sphere_big_1_20 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/hemi-sphere-big/params/DoLP',
        aolp_dir= root_dir + '/real-world/hemi-sphere-big/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/hemi-sphere-big/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/hemi-sphere-big/masks',
                                                                                          label_dir= root_dir + '/real-world/hemi-sphere-big/normals-png', transform=augs_train)
    dataset_hemi_sphere_small_1_20 = TransSfPDataset(
        dolp_dir= root_dir + '/real-world/hemi-sphere-small/params/DoLP',
        aolp_dir= root_dir + '/real-world/hemi-sphere-small/params/AoLP',
        synthesis_normals_dir= root_dir + '/real-world/hemi-sphere-small/synthesis-normals',
                                                                                          mask_dir = root_dir + '/real-world/hemi-sphere-small/masks',
                                                                                          label_dir= root_dir + '/real-world/hemi-sphere-small/normals-png', transform=augs_train)

    # synthetic datasets

    dataset_synthetic_polar_bun_zipper_back = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/bun-zipper-back/params/DoLP',
        aolp_dir= root_dir + '/synthetic/bun-zipper-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/bun-zipper-back/synthesis-normals',
                                                                              mask_dir= root_dir + '/synthetic/bun-zipper-back/masks',
                                                                              label_dir= root_dir + '/synthetic/bun-zipper-back/normals-png',
                                                                              transform=augs_train)
    dataset_synthetic_polar_bun_zipper_front = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/bun-zipper-front/params/DoLP',
        aolp_dir= root_dir + '/synthetic/bun-zipper-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/bun-zipper-front/synthesis-normals',
                                                                               mask_dir= root_dir + '/synthetic/bun-zipper-front/masks',
                                                                               label_dir= root_dir + '/synthetic/bun-zipper-front/normals-png',
                                                                               transform=augs_train)
    dataset_synthetic_polar_armadillo_back = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/armadillo-back/params/DoLP',
        aolp_dir= root_dir + '/synthetic/armadillo-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/armadillo-back/synthesis-normals',
                                                                             mask_dir= root_dir + '/synthetic/armadillo-back/masks',
                                                                             label_dir= root_dir + '/synthetic/armadillo-back/normals-png',
                                                                             transform=augs_train)
    dataset_synthetic_polar_armadillo_front = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/armadillo-front/params/DoLP',
        aolp_dir= root_dir + '/synthetic/armadillo-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/armadillo-front/synthesis-normals',
                                                                              mask_dir= root_dir + '/synthetic/armadillo-front/masks',
                                                                              label_dir= root_dir + '/synthetic/armadillo-front/normals-png',
                                                                              transform=augs_train)
    dataset_synthetic_polar_dragon_vrip = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/dragon-vrip/params/DoLP',
        aolp_dir= root_dir + '/synthetic/dragon-vrip/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/dragon-vrip/synthesis-normals',
                                                                          mask_dir= root_dir + '/synthetic/dragon-vrip/masks',
                                                                          label_dir= root_dir + '/synthetic/dragon-vrip/normals-png',
                                                                          transform=augs_train)
    dataset_synthetic_polar_happy_vrip_back= TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/happy-vrip-back/params/DoLP',
        aolp_dir= root_dir + '/synthetic/happy-vrip-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/happy-vrip-back/synthesis-normals',
                                                                             mask_dir= root_dir + '/synthetic/happy-vrip-back/masks',
                                                                             label_dir= root_dir + '/synthetic/happy-vrip-back/normals-png',
                                                                             transform=augs_train)
    dataset_synthetic_polar_happy_vrip_front= TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/happy-vrip-front/params/DoLP',
        aolp_dir= root_dir + '/synthetic/happy-vrip-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/happy-vrip-front/synthesis-normals',
                                                                              mask_dir= root_dir + '/synthetic/happy-vrip-front/masks',
                                                                              label_dir= root_dir + '/synthetic/happy-vrip-front/normals-png',
                                                                              transform=augs_train)
    dataset_synthetic_polar_middle_round_cup = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/middle-round-cup/params/DoLP',
        aolp_dir= root_dir + '/synthetic/middle-round-cup/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/middle-round-cup/synthesis-normals',
                                                                               mask_dir= root_dir + '/synthetic/middle-round-cup/masks',
                                                                               label_dir= root_dir + '/synthetic/middle-round-cup/normals-png',
                                                                               transform=augs_train)
    dataset_synthetic_polar_bear_front = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/bear-front/params/DoLP',
        aolp_dir= root_dir + '/synthetic/bear-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/bear-front/synthesis-normals',
                                                                         mask_dir= root_dir + '/synthetic/bear-front/masks',
                                                                         label_dir= root_dir + '/synthetic/bear-front/normals-png',
                                                                         transform=augs_train)
    dataset_synthetic_polar_cow_front = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/cow-front/params/DoLP',
        aolp_dir= root_dir + '/synthetic/cow-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/cow-front/synthesis-normals',
                                                                        mask_dir= root_dir + '/synthetic/cow-front/masks',
                                                                        label_dir= root_dir + '/synthetic/cow-front/normals-png',
                                                                        transform=augs_train)
    dataset_synthetic_polar_cow_back = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/cow-back/params/DoLP',
        aolp_dir= root_dir + '/synthetic/cow-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/cow-back/synthesis-normals',
                                                                       mask_dir= root_dir + '/synthetic/cow-back/masks',
                                                                       label_dir= root_dir + '/synthetic/cow-back/normals-png',
                                                                       transform=augs_train)
    dataset_synthetic_polar_pot_back = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/pot-back/params/DoLP',
        aolp_dir= root_dir + '/synthetic/pot-back/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/pot-back/synthesis-normals',
                                                                       mask_dir= root_dir + '/synthetic/pot-back/masks',
                                                                       label_dir= root_dir + '/synthetic/pot-back/normals-png',
                                                                       transform=augs_train)
    dataset_synthetic_polar_pot_front = TransSfPDataset(
        dolp_dir= root_dir + '/synthetic/pot-front/params/DoLP',
        aolp_dir= root_dir + '/synthetic/pot-front/params/AoLP',
        synthesis_normals_dir= root_dir + '/synthetic/pot-front/synthesis-normals',
                                                                       mask_dir= root_dir + '/synthetic/pot-front/masks',
                                                                       label_dir= root_dir + '/synthetic/pot-front/normals-png',
                                                                       transform=augs_train)



    db_list_synthetic  = [dataset_synthetic_polar_bun_zipper_back,dataset_synthetic_polar_bun_zipper_front,dataset_synthetic_polar_armadillo_back,dataset_synthetic_polar_armadillo_front,
                          dataset_synthetic_polar_dragon_vrip,dataset_synthetic_polar_happy_vrip_back,dataset_synthetic_polar_happy_vrip_front,
                          dataset_synthetic_polar_middle_round_cup,dataset_synthetic_polar_bear_front,dataset_synthetic_polar_cow_front,dataset_synthetic_polar_cow_back,
                          dataset_synthetic_polar_pot_back,dataset_synthetic_polar_pot_front]
    db_list_real = [dataset_middle_square_cup_black_background_12_28,dataset_middle_round_cup_black_background_12_28,dataset_middle_white_cup_black_background_12_28]
    db_train_list = db_list_synthetic + db_list_real


    dataset = torch.utils.data.ConcatDataset(db_train_list)

    #-- 2、create dataloader
    trainLoader = DataLoader(dataset,
                             num_workers=num_workers,
                             batch_size=batch_size,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_tiny_white_cup = DataLoader(dataset_tiny_white_cup_black_background_12_28,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_tiny_white_cup_edges = DataLoader(dataset_tiny_white_cup_edges_black_background_12_28,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_bird_back = DataLoader(dataset_bird_back_1_20,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_cat_back = DataLoader(dataset_cat_back_1_20,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_cat_front = DataLoader(dataset_cat_front_1_20,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_hemi_sphere_big = DataLoader(dataset_hemi_sphere_big_1_20,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)
    testLoader_hemi_sphere_small = DataLoader(dataset_hemi_sphere_small_1_20,
                             batch_size=1,
                             num_workers=num_workers,
                             drop_last=True,
                             pin_memory=pin_memory)



    print("trainLoader size:",trainLoader.__len__()*trainLoader.batch_size)
    return trainLoader,testLoader_tiny_white_cup,testLoader_tiny_white_cup_edges,testLoader_bird_back,testLoader_cat_back,testLoader_cat_front,testLoader_hemi_sphere_big,testLoader_hemi_sphere_small

