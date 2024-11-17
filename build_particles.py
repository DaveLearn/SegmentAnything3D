
from dataclasses import dataclass
from pathlib import Path
import time
from psdstaticdataset import StaticDataset
from initializerdefs import SceneSetup
import logging
import tyro

from particle_builder import initialize_scene


@dataclass
class Args:

    dataset_path: tyro.conf.Positional[Path]
    """
        Path to the dataset.
    """
    scene_path: tyro.conf.Positional[Path]
    """
        Path to the scene info. (a pickled SceneSetup).
    """
    

def run():
    # setup logging
    logger = logging.getLogger("sam3d-builder")
    logger.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    args = tyro.cli(Args)
    
    # start initialization
    logger.info("--------------")
    logger.info("Starting initialization")
    logger.info(f"params: {args}")

    logger.info(f"Loading dataset from {args.dataset_path}...")
    dataset = StaticDataset(args.dataset_path)
    logger.info("Dataset loaded.")

    logger.info(f"Loading scene setup from {args.scene_path}...")
    scene = SceneSetup.load(args.scene_path)
    logger.info("Scene loaded.")

    logger.info("Initializing scene...")

    project_root = Path(__file__).parent
    output_dir = project_root / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset.base_path.name}"
    
 
    objects = initialize_scene(dataset, scene, intermediate_outputs_path=output_dir)
    logger.info(f"Scene initialized with {len(objects.objects)} objects.")

    logger.info("Saving objects...")
    
    
    output_path = output_dir / "objectsdef.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    objects.save(output_path)

    logger.info(f"Objects saved to {output_path}")
    # output path in format expected by caller
    print(f"objects_path: {output_path}")

if __name__ == "__main__":
    run()
