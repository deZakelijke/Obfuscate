import time
import argparse
import os
from statistics import write_multiple_csv
from facenet.facenet_benchmark_oo import FacenetBenchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default="/home/douwe/Documents/MultiePIE_all_data/",
                        help='directory with crowd images')
    parser.add_argument('--source_filename', type=str, default='results/FaceNet_data',
                        help='prefix for the storing of the embeddings')
    parser.add_argument('--augmented_dir', type=str, default=None,
                        help='Directory of augmeted target images')
    parser.add_argument('--number_of_targets', type=int, default=50,
                        help='this will determine the number of unique ids')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--base_threshold', type=float, default=0.7,
                        help='First threshold value to try')
    parser.add_argument('--threshold_range', type=int, default=1,
                        help='Numer of threshold values to try')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='this will determine name for the folder where the generated \
                        csv output files will be written.')
    parser.add_argument('--dataset', type=str, default='MultiPIE',
                        help='Name of the dataset, must be MultiPIE of RafD')
    args, unparsed = parser.parse_known_args()

    if args.dataset == "MultiPIE":
        from dataloader import *
    elif args.dataset == "RafD":
        from dataloader_rafd import *
    else:
        raise ValueError("Illegal dataset")

    args.experiment_name = f"{args.experiment_name}_{args.dataset}_threshold_{str(args.base_threshold).replace('.', '-')}"
    if not os.path.exists(f"results/{args.experiment_name}"):
        os.makedirs(f"results/{args.experiment_name}")
    else:
        for f in os.listdir(f"results/{args.experiment_name}"):
            file_path = os.path.join(f"results/{args.experiment_name}", f)
            os.unlink(file_path)

    seed = int(time.time())

    # THIS SHOULD CONTAIN MORE VARIATIONS
    if args.dataset == "MultiPIE":
        mode = Mode(number_of_targets=args.number_of_targets,
                    expression=[NEUTRAL],
                    viewpoint=[VN, VL2, VR2, VL3, VR2],
                    illumination=[IALL],
                    pitch=[P0],
                    yaw=[Y0],
                    roll=[R0],
                    illum_change=[ILN],
                    illum_intensity=[INN],
                    glasses=[])
    if args.dataset == "RafD":
        mode = ModeRafd(number_of_targets=args.number_of_targets,
                        viewpoint = [VN],
                        ethnicity = [KID, CAUCASIAN, MOROCCAN],
                        expression = [NEUTRAL],
                        gaze = [FRONTAL])

    source_dlr, target_dlr, target_classes = get_dataloader(args.source_dir, mode=mode,
                                                            augmented_files=args.augmented_dir,
                                                            batch_size=args.batch_size, seed=seed)
    print(len(target_dlr))
    print("saving tmp csv with sources")
    save_sources(source_dlr)

    print("RUNNING LARGE EXPERIMENT")
    fb = FacenetBenchmark(source_dlr, target_dlr,
                          source_filename=args.source_filename,
                          experiment_name=args.experiment_name,
                          base_threshold=args.base_threshold,
                          threshold_range=args.threshold_range,
                          experiment_extension="variations")
    fb.classify(target_classes)

    # THIS SHOULD CONTAIN LESS VARIATIONS
    if args.dataset == "MultiPIE":
        mode = Mode(number_of_targets=args.number_of_targets,
                     expression=[NEUTRAL],
                     viewpoint=[VL1],
                     illumination=[IALL],
                     pitch=[P0],
                     yaw=[Y0],
                     roll=[R0],
                     illum_change=[ILN],
                     illum_intensity=[INN],
                     glasses=[])
    if args.dataset == "RafD":
        mode = ModeRafd(number_of_targets=args.number_of_targets,
                        viewpoint = [VN],
                        ethnicity = [KID, CAUCASIAN, MOROCCAN],
                        expression = [NEUTRAL],
                        gaze = [FRONTAL])

    # dataloader that overwrites the sources to have consistent sets
    source_dlr, target_dlr, target_classes = get_dataloader(args.source_dir, mode=mode,
                                                            batch_size=args.batch_size,
                                                            load_saved_sources=True, seed=seed)
    print("RUNNING SMALLER EXPERIMENT")
    fb = FacenetBenchmark(source_dlr, target_dlr,
                          source_filename=args.source_filename,
                          experiment_name=args.experiment_name,
                          base_threshold=args.base_threshold,
                          threshold_range=args.threshold_range,
                          experiment_extension="base")

    fb.classify(target_classes)
    
    # THIS SHOULD CONTAIN AUGMENTATIONS
    if args.dataset == "MultiPIE":
        mode = Mode(number_of_targets=args.number_of_targets,
                     expression=[NEUTRAL],
                     viewpoint=[VN],
                     illumination=[IALL],
                     pitch=[P0],
                     yaw=[Y0, Y30,Y_30, Y_45, Y45],
                     roll=[R0],
                     illum_change=[ILN],
                     illum_intensity=[INN],
                     glasses=[])
    if args.dataset == "RafD":
        mode = ModeRafd(number_of_targets=args.number_of_targets,
                        viewpoint = [VN],
                        ethnicity = [KID, CAUCASIAN, MOROCCAN],
                        expression = [NEUTRAL],
                        gaze = [FRONTAL])

    source_dlr, target_dlr, target_classes = get_dataloader(args.source_dir, mode=mode,
                                                            augmented_files=args.augmented_dir,
                                                            load_saved_sources=True,
                                                            batch_size=args.batch_size, seed=seed)

    print("RUNNING AUGMENTATION EXPERIMENT")
    fb = FacenetBenchmark(source_dlr, target_dlr,
                          source_filename=args.source_filename,
                          experiment_name=args.experiment_name,
                          base_threshold=args.base_threshold,
                          threshold_range=args.threshold_range,
                          experiment_extension="augmentation")
    fb.classify(target_classes)
    
    print("removing tmp csv")
    remove_sources()

    print("Making results plot")
    rafd = False
    if args.dataset == "RafD":
        rafd = True
    write_multiple_csv(f"results/{args.experiment_name}", "stats.md", rafd=rafd)
