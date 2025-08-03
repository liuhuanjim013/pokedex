#!/usr/bin/env python3
"""
Data preprocessing script for Pokemon classifier.
Prepares raw images for YOLO training format.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pokemon name mapping for numbered directories
POKEMON_NAMES = {
    # Generation 1 (001-151)
    "0001": "bulbasaur", "0002": "ivysaur", "0003": "venusaur", "0004": "charmander", "0005": "charmeleon",
    "0006": "charizard", "0007": "squirtle", "0008": "wartortle", "0009": "blastoise", "0010": "caterpie",
    "0011": "metapod", "0012": "butterfree", "0013": "weedle", "0014": "kakuna", "0015": "beedrill",
    "0016": "pidgey", "0017": "pidgeotto", "0018": "pidgeot", "0019": "rattata", "0020": "raticate",
    "0021": "spearow", "0022": "fearow", "0023": "ekans", "0024": "arbok", "0025": "pikachu",
    "0026": "raichu", "0027": "sandshrew", "0028": "sandslash", "0029": "nidoran-f", "0030": "nidorina",
    "0031": "nidoqueen", "0032": "nidoran-m", "0033": "nidorino", "0034": "nidoking", "0035": "clefairy",
    "0036": "clefable", "0037": "vulpix", "0038": "ninetales", "0039": "jigglypuff", "0040": "wigglytuff",
    "0041": "zubat", "0042": "golbat", "0043": "oddish", "0044": "gloom", "0045": "vileplume",
    "0046": "paras", "0047": "parasect", "0048": "venonat", "0049": "venomoth", "0050": "diglett",
    "0051": "dugtrio", "0052": "meowth", "0053": "persian", "0054": "psyduck", "0055": "golduck",
    "0056": "mankey", "0057": "primeape", "0058": "growlithe", "0059": "arcanine", "0060": "poliwag",
    "0061": "poliwhirl", "0062": "poliwrath", "0063": "abra", "0064": "kadabra", "0065": "alakazam",
    "0066": "machop", "0067": "machoke", "0068": "machamp", "0069": "bellsprout", "0070": "weepinbell",
    "0071": "victreebel", "0072": "tentacool", "0073": "tentacruel", "0074": "geodude", "0075": "graveler",
    "0076": "golem", "0077": "ponyta", "0078": "rapidash", "0079": "slowpoke", "0080": "slowbro",
    "0081": "magnemite", "0082": "magneton", "0083": "farfetchd", "0084": "doduo", "0085": "dodrio",
    "0086": "seel", "0087": "dewgong", "0088": "grimer", "0089": "muk", "0090": "shellder",
    "0091": "cloyster", "0092": "gastly", "0093": "haunter", "0094": "gengar", "0095": "onix",
    "0096": "drowzee", "0097": "hypno", "0098": "krabby", "0099": "kingler", "0100": "voltorb",
    "0101": "electrode", "0102": "exeggcute", "0103": "exeggutor", "0104": "cubone", "0105": "marowak",
    "0106": "hitmonlee", "0107": "hitmonchan", "0108": "lickitung", "0109": "koffing", "0110": "weezing",
    "0111": "rhyhorn", "0112": "rhydon", "0113": "chansey", "0114": "tangela", "0115": "kangaskhan",
    "0116": "horsea", "0117": "seadra", "0118": "goldeen", "0119": "seaking", "0120": "staryu",
    "0121": "starmie", "0122": "mr-mime", "0123": "scyther", "0124": "jynx", "0125": "electabuzz",
    "0126": "magmar", "0127": "pinsir", "0128": "tauros", "0129": "magikarp", "0130": "gyarados",
    "0131": "lapras", "0132": "ditto", "0133": "eevee", "0134": "vaporeon", "0135": "jolteon",
    "0136": "flareon", "0137": "porygon", "0138": "omanyte", "0139": "omastar", "0140": "kabuto",
    "0141": "kabutops", "0142": "aerodactyl", "0143": "snorlax", "0144": "articuno", "0145": "zapdos",
    "0146": "moltres", "0147": "dratini", "0148": "dragonair", "0149": "dragonite", "0150": "mewtwo",
    "0151": "mew",
    # Generation 2 (152-251)
    "0152": "chikorita", "0153": "bayleef", "0154": "meganium", "0155": "cyndaquil", "0156": "quilava",
    "0157": "typhlosion", "0158": "totodile", "0159": "croconaw", "0160": "feraligatr", "0161": "sentret",
    "0162": "furret", "0163": "hoothoot", "0164": "noctowl", "0165": "ledyba", "0166": "ledian",
    "0167": "spinarak", "0168": "ariados", "0169": "crobat", "0170": "chinchou", "0171": "lanturn",
    "0172": "pichu", "0173": "cleffa", "0174": "igglybuff", "0175": "togepi", "0176": "togetic",
    "0177": "natu", "0178": "xatu", "0179": "mareep", "0180": "flaaffy", "0181": "ampharos",
    "0182": "bellossom", "0183": "marill", "0184": "azumarill", "0185": "sudowoodo", "0186": "politoed",
    "0187": "hoppip", "0188": "skiploom", "0189": "jumpluff", "0190": "aipom", "0191": "sunkern",
    "0192": "sunflora", "0193": "yanma", "0194": "wooper", "0195": "quagsire", "0196": "espeon",
    "0197": "umbreon", "0198": "murkrow", "0199": "slowking", "0200": "misdreavus", "0201": "unown",
    "0202": "wobbuffet", "0203": "girafarig", "0204": "pineco", "0205": "forretress", "0206": "dunsparce",
    "0207": "gligar", "0208": "steelix", "0209": "snubbull", "0210": "granbull", "0211": "qwilfish",
    "0212": "scizor", "0213": "shuckle", "0214": "heracross", "0215": "sneasel", "0216": "teddiursa",
    "0217": "ursaring", "0218": "slugma", "0219": "magcargo", "0220": "swinub", "0221": "piloswine",
    "0222": "corsola", "0223": "remoraid", "0224": "octillery", "0225": "delibird", "0226": "mantine",
    "0227": "skarmory", "0228": "houndour", "0229": "houndoom", "0230": "kingdra", "0231": "phanpy",
    "0232": "donphan", "0233": "porygon2", "0234": "stantler", "0235": "smeargle", "0236": "tyrogue",
    "0237": "hitmontop", "0238": "smoochum", "0239": "elekid", "0240": "magby", "0241": "miltank",
    "0242": "blissey", "0243": "raikou", "0244": "entei", "0245": "suicune", "0246": "larvitar",
    "0247": "pupitar", "0248": "tyranitar", "0249": "lugia", "0250": "ho-oh", "0251": "celebi",
    # Generation 3 (252-386)
    "0252": "treecko", "0253": "grovyle", "0254": "sceptile", "0255": "torchic", "0256": "combusken",
    "0257": "blaziken", "0258": "mudkip", "0259": "marshtomp", "0260": "swampert", "0261": "poochyena",
    "0262": "mightyena", "0263": "zigzagoon", "0264": "linoone", "0265": "wurmple", "0266": "silcoon",
    "0267": "beautifly", "0268": "cascoon", "0269": "dustox", "0270": "lotad", "0271": "lombre",
    "0272": "ludicolo", "0273": "seedot", "0274": "nuzleaf", "0275": "shiftry", "0276": "taillow",
    "0277": "swellow", "0278": "wingull", "0279": "pelipper", "0280": "ralts", "0281": "kirlia",
    "0282": "gardevoir", "0283": "surskit", "0284": "masquerain", "0285": "shroomish", "0286": "breloom",
    "0287": "slakoth", "0288": "vigoroth", "0289": "slaking", "0290": "nincada", "0291": "ninjask",
    "0292": "shedinja", "0293": "whismur", "0294": "loudred", "0295": "exploud", "0296": "makuhita",
    "0297": "hariyama", "0298": "azurill", "0299": "nosepass", "0300": "skitty", "0301": "delcatty",
    "0302": "sableye", "0303": "mawile", "0304": "aron", "0305": "lairon", "0306": "aggron",
    "0307": "meditite", "0308": "medicham", "0309": "electrike", "0310": "manectric", "0311": "plusle",
    "0312": "minun", "0313": "volbeat", "0314": "illumise", "0315": "roselia", "0316": "gulpin",
    "0317": "swalot", "0318": "carvanha", "0319": "sharpedo", "0320": "wailmer", "0321": "wailord",
    "0322": "numel", "0323": "camerupt", "0324": "torkoal", "0325": "spoink", "0326": "grumpig",
    "0327": "spinda", "0328": "trapinch", "0329": "vibrava", "0330": "flygon", "0331": "cacnea",
    "0332": "cacturne", "0333": "swablu", "0334": "altaria", "0335": "zangoose", "0336": "seviper",
    "0337": "lunatone", "0338": "solrock", "0339": "barboach", "0340": "whiscash", "0341": "corphish",
    "0342": "crawdaunt", "0343": "baltoy", "0344": "claydol", "0345": "lileep", "0346": "cradily",
    "0347": "anorith", "0348": "armaldo", "0349": "feebas", "0350": "milotic", "0351": "castform",
    "0352": "kecleon", "0353": "shuppet", "0354": "banette", "0355": "duskull", "0356": "dusclops",
    "0357": "tropius", "0358": "chimecho", "0359": "absol", "0360": "wynaut", "0361": "snorunt",
    "0362": "glalie", "0363": "spheal", "0364": "sealeo", "0365": "walrein", "0366": "clamperl",
    "0367": "huntail", "0368": "gorebyss", "0369": "relicanth", "0370": "luvdisc", "0371": "bagon",
    "0372": "shelgon", "0373": "salamence", "0374": "beldum", "0375": "metang", "0376": "metagross",
    "0377": "regirock", "0378": "regice", "0379": "registeel", "0380": "latias", "0381": "latios",
    "0382": "kyogre", "0383": "groudon", "0384": "rayquaza", "0385": "jirachi", "0386": "deoxys"
}

class PokemonDataPreprocessor:
    """Preprocess Pokemon images for YOLO training."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.image_size = self.config['processing']['image_size']
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "images").mkdir(exist_ok=True)
        (self.processed_dir / "metadata").mkdir(exist_ok=True)
    
    def process_gen1_3_dataset(self, dataset_path: str) -> Dict:
        """
        Process the 900MB gen1-3 dataset with numbered directories.
        
        Args:
            dataset_path: Path to the downloaded dataset
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing dataset from: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        # Expected structure:
        # dataset_path/
        #   ├── 0001/  # Bulbasaur
        #   │   ├── 0001bulbasaur-0.jpg
        #   │   ├── 0001Bulbasaur29.jpg
        #   │   └── ...
        #   ├── 0002/  # Ivysaur
        #   │   └── ...
        #   └── ...
        
        pokemon_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        pokemon_dirs.sort(key=lambda x: int(x.name))  # Sort by number
        logger.info(f"Found {len(pokemon_dirs)} Pokemon directories")
        
        processed_images = []
        pokemon_names = []
        pokemon_ids = []
        
        for pokemon_dir in tqdm(pokemon_dirs, desc="Processing Pokemon"):
            pokemon_id = pokemon_dir.name
            pokemon_name = POKEMON_NAMES.get(pokemon_id, f"unknown_{pokemon_id}")
            
            # Process all images in this Pokemon's directory
            image_files = list(pokemon_dir.glob("*.jpg")) + list(pokemon_dir.glob("*.png"))
            
            for img_file in image_files:
                try:
                    # Load and preprocess image
                    processed_path = self._process_single_image(img_file, pokemon_name)
                    
                    if processed_path:
                        processed_images.append(str(processed_path))
                        pokemon_names.append(pokemon_name)
                        pokemon_ids.append(pokemon_id)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {img_file}: {e}")
                    continue
        
        # Create metadata
        metadata = {
            'image_paths': processed_images,
            'pokemon_names': pokemon_names,
            'pokemon_ids': pokemon_ids,
            'total_images': len(processed_images),
            'unique_pokemon': len(set(pokemon_names)),
            'pokemon_mapping': POKEMON_NAMES
        }
        
        # Save metadata
        metadata_path = self.processed_dir / "metadata" / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save Pokemon mapping
        mapping_path = self.processed_dir / "metadata" / "pokemon_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(POKEMON_NAMES, f, indent=2)
        
        logger.info(f"Processed {len(processed_images)} images from {len(set(pokemon_names))} Pokemon")
        return metadata
    
    def _process_single_image(self, image_path: Path, pokemon_name: str) -> str:
        """
        Process a single image for YOLO training.
        
        Args:
            image_path: Path to input image
            pokemon_name: Name of the Pokemon
            
        Returns:
            Path to processed image
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to YOLO standard size
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Save processed image
        output_dir = self.processed_dir / "images" / pokemon_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{image_path.stem}_processed.jpg"
        cv2.imwrite(str(output_path), (img_normalized * 255).astype(np.uint8))
        
        return str(output_path)
    
    def create_yolo_dataset(self, output_dir: str = "data/processed/yolo_dataset"):
        """
        Create YOLO-compatible dataset structure.
        
        Args:
            output_dir: Output directory for YOLO dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO dataset structure
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_path = self.processed_dir / "metadata" / "dataset_info.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create Pokemon name to class ID mapping
        unique_pokemon = list(set(metadata['pokemon_names']))
        unique_pokemon.sort()  # Sort for consistent ordering
        pokemon_to_id = {name: idx for idx, name in enumerate(unique_pokemon)}
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # Group by Pokemon to ensure balanced splits
        pokemon_groups = {}
        for img_path, pokemon_name in zip(metadata['image_paths'], metadata['pokemon_names']):
            if pokemon_name not in pokemon_groups:
                pokemon_groups[pokemon_name] = []
            pokemon_groups[pokemon_name].append(img_path)
        
        train_images, temp_images = train_test_split(
            list(pokemon_groups.items()), 
            test_size=0.3, 
            random_state=42
        )
        
        val_images, test_images = train_test_split(
            temp_images, 
            test_size=0.5, 
            random_state=42
        )
        
        # Process splits
        self._process_split(train_images, output_dir, "train", pokemon_to_id)
        self._process_split(val_images, output_dir, "val", pokemon_to_id)
        self._process_split(test_images, output_dir, "test", pokemon_to_id)
        
        # Save class mapping
        class_mapping = {idx: name for name, idx in pokemon_to_id.items()}
        with open(output_dir / "classes.txt", 'w') as f:
            for idx in range(len(class_mapping)):
                f.write(f"{class_mapping[idx]}\n")
        
        logger.info(f"Created YOLO dataset at {output_dir}")
        logger.info(f"Classes: {len(class_mapping)}")
    
    def _process_split(self, split_data, output_dir, split_name, pokemon_to_id):
        """Process a single data split."""
        for pokemon_name, image_paths in split_data:
            class_id = pokemon_to_id[pokemon_name]
            
            for img_path in image_paths:
                # Copy image
                src_path = Path(img_path)
                dst_path = output_dir / "images" / split_name / src_path.name
                
                import shutil
                shutil.copy2(src_path, dst_path)
                
                # Create YOLO label file (classification format)
                label_path = output_dir / "labels" / split_name / f"{src_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"{class_id}\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Pokemon dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to raw dataset")
    parser.add_argument("--config", default="configs/data_config.yaml", help="Config file path")
    parser.add_argument("--create_yolo_dataset", action="store_true", help="Create YOLO dataset structure")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = PokemonDataPreprocessor(args.config)
    
    # Process dataset
    metadata = preprocessor.process_gen1_3_dataset(args.dataset_path)
    
    # Create YOLO dataset if requested
    if args.create_yolo_dataset:
        preprocessor.create_yolo_dataset()
    
    print(f"Processing complete! Processed {metadata['total_images']} images from {metadata['unique_pokemon']} Pokemon")

if __name__ == "__main__":
    main() 