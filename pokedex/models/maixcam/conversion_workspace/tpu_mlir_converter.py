#!/usr/bin/env python3
"""
TPU-MLIR Conversion Script for Pokemon Classifier
Converts ONNX model to MaixCam compatible .cvimodel format
"""

import os
import sys
import subprocess
import glob
import psutil
import random
import shutil
from pathlib import Path
from collections import defaultdict

def get_memory_usage():
    """Get current memory usage in GB"""
    memory = psutil.virtual_memory()
    return memory.used / (1024**3), memory.total / (1024**3)

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🚀 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    
    # Check memory before command
    used_gb, total_gb = get_memory_usage()
    print(f"   Memory before: {used_gb:.1f}GB / {total_gb:.1f}GB")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        
        # Check memory after command
        used_gb, total_gb = get_memory_usage()
        print(f"   Memory after: {used_gb:.1f}GB / {total_gb:.1f}GB")
        
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        if e.stdout:
            print(f"   Stdout: {e.stdout.strip()}")
        print(f"   Return code: {e.returncode}")
        return False

def calculate_optimal_calibration_images(total_images, available_memory_gb):
    """Calculate optimal number of calibration images based on available memory"""
    # Conservative estimate: each image ~0.1MB in memory during calibration
    # With 16GB total, we can safely use 8-10GB for calibration
    max_memory_for_calibration = min(available_memory_gb * 0.6, 10)  # Use 60% of available memory, max 10GB
    
    # Estimate images per GB (conservative)
    images_per_gb = 8000  # Conservative estimate
    
    optimal_images = int(max_memory_for_calibration * images_per_gb)
    
    # Ensure we don't exceed total available images
    optimal_images = min(optimal_images, total_images)
    
    # Set reasonable bounds
    min_images = 1000
    max_images = 15000  # Conservative upper limit
    
    optimal_images = max(min_images, min(optimal_images, max_images))
    
    return optimal_images

def create_stratified_calibration_dataset(total_calibration_images, images_dir="images"):
    """
    Create a stratified calibration dataset ensuring equal representation of all 1025 Pokemon classes.
    
    Args:
        total_calibration_images: Total number of images to use for calibration
        images_dir: Directory containing the images
    
    Returns:
        str: Path to the calibration images directory
    """
    print(f"🎯 Creating stratified calibration dataset...")
    print(f"   Target: {total_calibration_images:,} images")
    print(f"   Classes: 1025 Pokemon")
    
    # Get all image files
    image_files = glob.glob(f"{images_dir}/*.jpg")
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return None
    
    print(f"📸 Found {len(image_files):,} total images")
    
    # Group images by Pokemon class (extract class ID from filename)
    class_images = defaultdict(list)
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Expected format: {pokemon_id:04d}_{image_number:03d}.jpg
        # Example: 0001_001.jpg, 0002_001.jpg, etc.
        # Pokemon ID 0001 → class_id 0 (Bulbasaur)
        # Pokemon ID 0002 → class_id 1 (Ivysaur)
        # Pokemon ID 1025 → class_id 1024 (Last Pokemon)
        try:
            pokemon_id_str = filename.split('_')[0]
            pokemon_id = int(pokemon_id_str)
            # Convert Pokemon ID to 0-based class ID
            # Pokemon ID 0001 → class_id 0
            # Pokemon ID 0002 → class_id 1
            # Pokemon ID 1025 → class_id 1024
            class_id = pokemon_id - 1
            class_images[class_id].append(img_path)
        except (ValueError, IndexError):
            print(f"⚠️  Skipping file with unexpected format: {filename}")
            continue
    
    print(f"📊 Found images for {len(class_images)} Pokemon classes")
    
    # Check if we have all 1025 classes
    missing_classes = set(range(1025)) - set(class_images.keys())
    if missing_classes:
        print(f"⚠️  Warning: Missing images for {len(missing_classes)} classes: {sorted(missing_classes)[:10]}...")
    
    # Calculate images per class for balanced representation
    images_per_class = total_calibration_images // 1025
    remainder = total_calibration_images % 1025
    
    print(f"📈 Target: {images_per_class} images per class")
    print(f"📈 Extra: {remainder} images distributed across classes")
    
    # Create calibration directory
    calibration_dir = "calibration_images"
    if os.path.exists(calibration_dir):
        shutil.rmtree(calibration_dir)
    os.makedirs(calibration_dir)
    
    # Select images for each class
    selected_images = []
    classes_with_insufficient_images = []
    
    for class_id in range(1025):
        available_images = class_images.get(class_id, [])
        target_count = images_per_class + (1 if class_id < remainder else 0)
        
        if len(available_images) >= target_count:
            # Randomly sample target_count images from this class
            selected = random.sample(available_images, target_count)
            selected_images.extend(selected)
        else:
            # Use all available images for this class
            selected_images.extend(available_images)
            classes_with_insufficient_images.append(class_id)
            print(f"⚠️  Class {class_id}: Only {len(available_images)} images available (wanted {target_count})")
    
    print(f"📊 Selected {len(selected_images):,} images for calibration")
    
    if classes_with_insufficient_images:
        print(f"⚠️  {len(classes_with_insufficient_images)} classes have insufficient images")
        print(f"   Classes: {sorted(classes_with_insufficient_images)[:10]}...")
    
    # Copy selected images to calibration directory
    print(f"📁 Copying images to {calibration_dir}...")
    for i, img_path in enumerate(selected_images):
        if i % 1000 == 0:
            print(f"   Progress: {i:,}/{len(selected_images):,}")
        
        # Copy with original filename to preserve class information
        filename = os.path.basename(img_path)
        dest_path = os.path.join(calibration_dir, filename)
        shutil.copy2(img_path, dest_path)
    
    print(f"✅ Stratified calibration dataset created: {calibration_dir}")
    print(f"   Total images: {len(selected_images):,}")
    print(f"   Classes represented: {len(set(int(os.path.basename(f).split('_')[0]) for f in selected_images))}")
    
    return calibration_dir

def main():
    print("🎯 TPU-MLIR Conversion for Pokemon Classifier")
    print("=" * 50)
    
    # Configuration
    model_name = "pokemon_classifier"
    onnx_model = f"{model_name}.onnx"
    
    # Determine workspace directory based on environment
    if os.path.exists("/workspace"):
        # Docker container environment
        workspace_dir = "/workspace"
    else:
        # Local environment (fallback)
        workspace_dir = "."
    
    # Change to workspace directory
    os.chdir(workspace_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check if ONNX model exists
    if not os.path.exists(onnx_model):
        print(f"❌ ONNX model not found: {onnx_model}")
        return False
    
    print(f"📊 Model Information:")
    print(f"  - Model: {onnx_model}")
    print(f"  - Input size: 256x256")
    print(f"  - Classes: 1025 Pokemon")
    
    # Step 1: Transform ONNX to MLIR
    step1_cmd = [
        "model_transform.py",
        "--model_name", model_name,
        "--model_def", onnx_model,
        "--input_shapes", "[[1,3,256,256]]",
        "--mean", "0,0,0",
        "--scale", "0.00392156862745098,0.00392156862745098,0.00392156862745098",
        "--pixel_format", "rgb",
        "--mlir", f"{model_name}.mlir"
    ]
    
    if not run_command(step1_cmd, "Step 1: Transforming ONNX to MLIR"):
        return False
    
    # Check if MLIR file was created
    mlir_file = f"{model_name}.mlir"
    if not os.path.exists(mlir_file):
        print(f"❌ MLIR file not created: {mlir_file}")
        return False
    
    print(f"✅ MLIR file created: {mlir_file}")
    
    # Step 2: Run calibration for INT8 quantization
    # Count images for calibration
    image_files = glob.glob("images/*.jpg")
    num_images = len(image_files)
    print(f"📸 Found {num_images:,} total images")
    
    # Calculate optimal number of calibration images based on available memory
    used_gb, total_gb = get_memory_usage()
    available_gb = total_gb - used_gb
    print(f"💾 Memory status: {used_gb:.1f}GB used, {available_gb:.1f}GB available")
    
    calibration_images = calculate_optimal_calibration_images(num_images, available_gb)
    print(f"📸 Using {calibration_images:,} images for calibration (optimized for memory)")
    
    # Create stratified calibration dataset ensuring equal class representation
    calibration_dir = create_stratified_calibration_dataset(calibration_images, "images")
    if not calibration_dir:
        print("❌ Failed to create stratified calibration dataset")
        return False
    
    # Use stratified calibration dataset
    step2_cmd = [
        "run_calibration.py",
        f"{model_name}.mlir",
        "--dataset", calibration_dir,
        "--input_num", str(calibration_images),
        "-o", f"{model_name}_cali_table"
    ]
    
    if not run_command(step2_cmd, "Step 2: Running calibration for INT8 quantization"):
        return False
    
    # Check if calibration table was created
    cali_table = f"{model_name}_cali_table"
    if not os.path.exists(cali_table):
        print(f"❌ Calibration table not created: {cali_table}")
        return False
    
    print(f"✅ Calibration table created: {cali_table}")
    
    # Step 3: Quantize to INT8
    step3_cmd = [
        "model_deploy.py",
        "--mlir", f"{model_name}.mlir",
        "--quantize", "INT8",
        "--calibration_table", cali_table,
        "--processor", "cv181x",
        "--model", f"{model_name}_int8.cvimodel"
    ]
    
    if not run_command(step3_cmd, "Step 3: Quantizing to INT8"):
        return False
    
    # Check if cvimodel was created
    cvimodel_file = f"{model_name}_int8.cvimodel"
    if not os.path.exists(cvimodel_file):
        print(f"❌ CVIModel file not created: {cvimodel_file}")
        return False
    
    print(f"✅ CVIModel file created: {cvimodel_file}")
    
    # Step 4: Create MUD file
    print("📝 Creating MUD file...")
    
    # Pokemon class names (all 1025 Pokemon)
    # Complete list of all 1025 Pokemon (generations 1-9) from existing config
    pokemon_names = [
        "bulbasaur", "ivysaur", "venusaur", "charmander", "charmeleon", "charizard",
        "squirtle", "wartortle", "blastoise", "caterpie", "metapod", "butterfree",
        "weedle", "kakuna", "beedrill", "pidgey", "pidgeotto", "pidgeot", "rattata",
        "raticate", "spearow", "fearow", "ekans", "arbok", "pikachu", "raichu",
        "sandshrew", "sandslash", "nidoran-f", "nidorina", "nidoqueen", "nidoran-m",
        "nidorino", "nidoking", "clefairy", "clefable", "vulpix", "ninetales",
        "jigglypuff", "wigglytuff", "zubat", "golbat", "oddish", "gloom", "vileplume",
        "paras", "parasect", "venonat", "venomoth", "diglett", "dugtrio", "meowth",
        "persian", "psyduck", "golduck", "mankey", "primeape", "growlithe", "arcanine",
        "poliwag", "poliwhirl", "poliwrath", "abra", "kadabra", "alakazam", "machop",
        "machoke", "machamp", "bellsprout", "weepinbell", "victreebel", "tentacool",
        "tentacruel", "geodude", "graveler", "golem", "ponyta", "rapidash", "slowpoke",
        "slowbro", "magnemite", "magneton", "farfetchd", "doduo", "dodrio", "seel",
        "dewgong", "grimer", "muk", "shellder", "cloyster", "gastly", "haunter",
        "gengar", "onix", "drowzee", "hypno", "krabby", "kingler", "voltorb",
        "electrode", "exeggcute", "exeggutor", "cubone", "marowak", "hitmonlee",
        "hitmonchan", "lickitung", "koffing", "weezing", "rhyhorn", "rhydon",
        "chansey", "tangela", "kangaskhan", "horsea", "seadra", "goldeen", "seaking",
        "staryu", "starmie", "mr-mime", "scyther", "jynx", "electabuzz", "magmar",
        "pinsir", "tauros", "magikarp", "gyarados", "lapras", "ditto", "eevee",
        "vaporeon", "jolteon", "flareon", "porygon", "omanyte", "omastar", "kabuto",
        "kabutops", "aerodactyl", "snorlax", "articuno", "zapdos", "moltres",
        "dratini", "dragonair", "dragonite", "mewtwo", "mew", "chikorita", "bayleef",
        "meganium", "cyndaquil", "quilava", "typhlosion", "totodile", "croconaw",
        "feraligatr", "sentret", "furret", "hoothoot", "noctowl", "ledyba", "ledian",
        "spinarak", "ariados", "crobat", "chinchou", "lanturn", "pichu", "cleffa",
        "igglybuff", "togepi", "togetic", "natu", "xatu", "mareep", "flaaffy",
        "ampharos", "bellossom", "marill", "azumarill", "sudowoodo", "politoed",
        "hoppip", "skiploom", "jumpluff", "aipom", "sunkern", "sunflora", "yanma",
        "wooper", "quagsire", "espeon", "umbreon", "murkrow", "slowking", "misdreavus",
        "unown", "wobbuffet", "girafarig", "pineco", "forretress", "dunsparce",
        "gligar", "steelix", "snubbull", "granbull", "qwilfish", "scizor", "shuckle",
        "heracross", "sneasel", "teddiursa", "ursaring", "slugma", "magcargo",
        "swinub", "piloswine", "corsola", "remoraid", "octillery", "delibird",
        "mantine", "skarmory", "houndour", "houndoom", "kingdra", "phanpy", "donphan",
        "porygon2", "stantler", "smeargle", "tyrogue", "hitmontop", "smoochum",
        "elekid", "magby", "miltank", "blissey", "raikou", "entei", "suicune",
        "larvitar", "pupitar", "tyranitar", "lugia", "ho-oh", "celebi", "treecko",
        "grovyle", "sceptile", "torchic", "combusken", "blaziken", "mudkip", "marshtomp",
        "swampert", "poochyena", "mightyena", "zigzagoon", "linoone", "wurmple",
        "silcoon", "beautifly", "cascoon", "dustox", "lotad", "lombre", "ludicolo",
        "seedot", "nuzleaf", "shiftry", "taillow", "swellow", "wingull", "pelipper",
        "ralts", "kirlia", "gardevoir", "surskit", "masquerain", "shroomish", "breloom",
        "slakoth", "vigoroth", "slaking", "nincada", "ninjask", "shedinja", "whismur",
        "loudred", "exploud", "makuhita", "hariyama", "azurill", "nosepass", "skitty",
        "delcatty", "sableye", "mawile", "aron", "lairon", "aggron", "meditite",
        "medicham", "electrike", "manectric", "plusle", "minun", "volbeat", "illumise",
        "roselia", "gulpin", "swalot", "carvanha", "sharpedo", "wailmer", "wailord",
        "numel", "camerupt", "torkoal", "spoink", "grumpig", "spinda", "trapinch",
        "vibrava", "flygon", "cacnea", "cacturne", "swablu", "altaria", "zangoose",
        "seviper", "lunatone", "solrock", "barboach", "whiscash", "corphish",
        "crawdaunt", "baltoy", "claydol", "lileep", "cradily", "anorith", "armaldo",
        "feebas", "milotic", "castform", "kecleon", "shuppet", "banette", "duskull",
        "dusclops", "tropius", "chimecho", "absol", "wynaut", "snorunt", "glalie",
        "spheal", "sealeo", "walrein", "clamperl", "huntail", "gorebyss", "relicanth",
        "luvdisc", "bagon", "shelgon", "salamence", "beldum", "metang", "metagross",
        "regirock", "regice", "registeel", "latias", "latios", "kyogre", "groudon",
        "rayquaza", "jirachi", "deoxys", "turtwig", "grotle", "torterra", "chimchar",
        "monferno", "infernape", "piplup", "prinplup", "empoleon", "starly", "staravia",
        "staraptor", "bidoof", "bibarel", "kricketot", "kricketune", "shinx", "luxio",
        "luxray", "budew", "roserade", "cranidos", "rampardos", "shieldon", "bastiodon",
        "burmy", "wormadam", "mothim", "combee", "vespiquen", "pachirisu", "buizel",
        "floatzel", "cherubi", "cherrim", "shellos", "gastrodon", "ambipom", "drifloon",
        "drifblim", "buneary", "lopunny", "mismagius", "honchkrow", "glameow", "purugly",
        "chingling", "stunky", "skuntank", "bronzor", "bronzong", "bonsly", "mime-jr",
        "happiny", "chatot", "spiritomb", "gible", "gabite", "garchomp", "munchlax",
        "riolu", "lucario", "hippopotas", "hippowdon", "skorupi", "drapion", "croagunk",
        "toxicroak", "carnivine", "finneon", "lumineon", "mantyke", "snover", "abomasnow",
        "weavile", "magnezone", "lickilicky", "rhyperior", "tangrowth", "electivire",
        "magmortar", "togekiss", "yanmega", "leafeon", "glaceon", "gliscor", "mamoswine",
        "porygon-z", "gallade", "probopass", "dusknoir", "froslass", "rotom", "uxie",
        "mesprit", "azelf", "dialga", "palkia", "heatran", "regigigas", "giratina",
        "cresselia", "phione", "manaphy", "darkrai", "shaymin", "arceus", "victini",
        "snivy", "servine", "serperior", "tepig", "pignite", "emboar", "oshawott",
        "dewott", "samurott", "patrat", "watchog", "lillipup", "herdier", "stoutland",
        "purrloin", "liepard", "pansage", "simisage", "pansear", "simisear", "panpour",
        "simipour", "munna", "musharna", "pidove", "tranquill", "unfezant", "blitzle",
        "zebstrika", "roggenrola", "boldore", "gigalith", "woobat", "swoobat", "drilbur",
        "excadrill", "audino", "timburr", "gurdurr", "conkeldurr", "tympole", "palpitoad",
        "seismitoad", "throh", "sawk", "sewaddle", "swadloon", "leavanny", "venipede",
        "whirlipede", "scolipede", "cottonee", "whimsicott", "petilil", "lilligant",
        "basculin", "sandile", "krokorok", "krookodile", "darumaka", "darmanitan",
        "maractus", "dwebble", "crustle", "scraggy", "scrafty", "sigilyph", "yamask",
        "cofagrigus", "tirtouga", "carracosta", "archen", "archeops", "trubbish",
        "garbodor", "zorua", "zoroark", "minccino", "cinccino", "gothita", "gothorita",
        "gothitelle", "solosis", "duosion", "reuniclus", "ducklett", "swanna",
        "vanillite", "vanillish", "vanilluxe", "deerling", "sawsbuck", "emolga",
        "karrablast", "escavalier", "foongus", "amoonguss", "frillish", "jellicent",
        "alomomola", "joltik", "galvantula", "ferroseed", "ferrothorn", "klink",
        "klang", "klinklang", "tynamo", "eelektrik", "eelektross", "elgyem", "beheeyem",
        "litwick", "lampent", "chandelure", "axew", "fraxure", "haxorus", "cubchoo",
        "beartic", "cryogonal", "shelmet", "accelgor", "stunfisk", "mienfoo", "mienshao",
        "druddigon", "golett", "golurk", "pawniard", "bisharp", "bouffalant", "rufflet",
        "braviary", "vullaby", "mandibuzz", "heatmor", "durant", "deino", "zweilous",
        "hydreigon", "larvesta", "volcarona", "cobalion", "terrakion", "virizion",
        "tornadus", "thundurus", "reshiram", "zekrom", "landorus", "kyurem", "keldeo",
        "meloetta", "genesect", "chespin", "quilladin", "chesnaught", "fennekin",
        "braixen", "delphox", "froakie", "frogadier", "greninja", "bunnelby", "diggersby",
        "fletchling", "fletchinder", "talonflame", "scatterbug", "spewpa", "vivillon",
        "litleo", "pyroar", "flabebe", "floette", "florges", "skiddo", "gogoat",
        "pancham", "pangoro", "furfrou", "espurr", "meowstic", "honedge", "doublade",
        "aegislash", "spritzee", "aromatisse", "swirlix", "slurpuff", "inkay", "malamar",
        "binacle", "barbaracle", "skrelp", "dragalge", "clauncher", "clawitzer",
        "helioptile", "heliolisk", "tyrunt", "tyrantrum", "amaura", "aurorus",
        "sylveon", "hawlucha", "dedenne", "carbink", "goomy", "sliggoo", "goodra",
        "klefki", "phantump", "trevenant", "pumpkaboo", "gourgeist", "bergmite",
        "avalugg", "noibat", "noivern", "xerneas", "yveltal", "zygarde", "diancie",
        "hoopa", "volcanion", "rowlet", "dartrix", "decidueye", "litten", "torracat",
        "incineroar", "popplio", "brionne", "primarina", "pikipek", "trumbeak",
        "toucannon", "yungoos", "gumshoos", "grubbin", "charjabug", "vikavolt",
        "crabrawler", "crabominable", "oricorio", "cutiefly", "ribombee", "rockruff",
        "lycanroc", "wishiwashi", "mareanie", "toxapex", "mudbray", "mudsdale",
        "dewpider", "araquanid", "fomantis", "lurantis", "morelull", "shiinotic",
        "salandit", "salazzle", "stufful", "bewear", "bounsweet", "steenee", "tsareena",
        "comfey", "oranguru", "passimian", "wimpod", "golisopod", "sandygast",
        "palossand", "pyukumuku", "type-null", "silvally", "minior", "komala",
        "turtonator", "togedemaru", "mimikyu", "bruxish", "drampa", "dhelmise",
        "jangmo-o", "hakamo-o", "kommo-o", "tapu-koko", "tapu-lele", "tapu-bulu",
        "tapu-fini", "cosmog", "cosmoem", "solgaleo", "lunala", "nihilego", "buzzwole",
        "pheromosa", "xurkitree", "celesteela", "kartana", "guzzlord", "necrozma",
        "magearna", "marshadow", "poipole", "naganadel", "stakataka", "blacephalon",
        "zeraora", "meltan", "melmetal", "grookey", "thwackey", "rillaboom",
        "scorbunny", "raboot", "cinderace", "sobble", "drizzile", "inteleon",
        "skwovet", "greedent", "rookidee", "corvisquire", "corviknight", "blipbug",
        "dottler", "orbeetle", "nickit", "thievul", "gossifleur", "eldegoss",
        "wooloo", "dubwool", "chewtle", "drednaw", "yamper", "boltund", "rolycoly",
        "carkol", "coalossal", "applin", "flapple", "appletun", "silicobra",
        "sandaconda", "cramorant", "arrokuda", "barraskewda", "toxel", "toxtricity",
        "sizzlipede", "centiskorch", "clobbopus", "grapploct", "sinistea", "polteageist",
        "hatenna", "hattrem", "hatterene", "impidimp", "morgrem", "grimmsnarl",
        "obstagoon", "perrserker", "cursola", "sirfetchd", "mr-rime", "runerigus",
        "milcery", "alcremie", "falinks", "pincurchin", "snom", "frosmoth",
        "stonjourner", "eiscue", "indeedee", "morpeko", "cufant", "copperajah",
        "dracozolt", "arctozolt", "dracovish", "arctovish", "duraludon", "dreepy",
        "drakloak", "dragapult", "zacian", "zamazenta", "eternatus", "kubfu",
        "urshifu", "zarude", "regieleki", "regidrago", "glastrier", "spectrier",
        "calyrex", "unknown-pokemon-0899", "unknown-pokemon-0900", "unknown-pokemon-0901",
        "unknown-pokemon-0902", "unknown-pokemon-0903", "unknown-pokemon-0904",
        "unknown-pokemon-0905", "sprigatito", "floragato", "meowscarada", "fuecoco",
        "crocalor", "skeledirge", "quaxly", "quaxwell", "quaquaval", "lechonk",
        "oinkologne", "tarountula", "spidops", "nymble", "lokix", "pawmi", "pawmo",
        "pawmot", "tandemaus", "maushold", "fidough", "dachsbun", "smoliv", "dolliv",
        "arboliva", "squawkabilly", "nacli", "naclstack", "garganacl", "charcadet",
        "armarouge", "ceruledge", "tadbulb", "bellibolt", "wattrel", "kilowattrel",
        "maschiff", "mabosstiff", "shroodle", "grafaiai", "bramblin", "brambleghast",
        "toedscool", "toedscruel", "klawf", "capsakid", "scovillain", "rellor",
        "rabsca", "flittle", "espathra", "tinkatink", "tinkatuff", "tinkaton",
        "wiglett", "wugtrio", "bombirdier", "finizen", "palafin", "varoom", "revavroom",
        "cyclizar", "orthworm", "glimmet", "glimmora", "greavard", "houndstone",
        "flamigo", "cetoddle", "cetitan", "veluza", "dondozo", "tatsugiri", "annihilape",
        "clodsire", "farigiraf", "dudunsparce", "kingambit", "great-tusk", "scream-tail",
        "brute-bonnet", "flutter-mane", "slither-wing", "sandy-shocks", "iron-treads",
        "iron-bundle", "iron-hands", "iron-jugulis", "iron-moth", "iron-thorns",
        "frigibax", "arctibax", "baxcalibur", "gimmighoul", "gholdengo", "wo-chien",
        "chien-pao", "ting-lu", "chi-yu", "roaring-moon", "iron-valiant", "koraidon",
        "miraidon", "walking-wake", "iron-leaves", "dipplin", "poltchageist", "sinistcha",
        "okidogi", "munkidori", "fezandipiti", "ogerpon", "archaludon", "hydrapple",
        "gouging-fire", "raging-bolt", "iron-boulder", "iron-crown", "terapagos",
        "pecharunt"
    ]
    
    # For brevity, I'll use a placeholder. In practice, you'd want the full list
    pokemon_classes_str = ",".join(pokemon_names)
    
    mud_content = f"""[basic]
type = cvimodel
model = {model_name}_int8.cvimodel

[extra]
model_type = yolov11
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
labels = {pokemon_classes_str}
"""
    
    mud_file = f"{model_name}.mud"
    with open(mud_file, 'w') as f:
        f.write(mud_content)
    
    print(f"✅ MUD file created: {mud_file}")
    
    # Final summary
    print("\n🎉 TPU-MLIR conversion completed successfully!")
    print("\n📁 Output files:")
    print(f"  - {cvimodel_file} (INT8 quantized model)")
    print(f"  - {mud_file} (Model description file)")
    print(f"  - {mlir_file} (Intermediate MLIR file)")
    print(f"  - {cali_table} (Calibration table)")
    
    # List created files
    print("\n📋 Created files:")
    for file in [cvimodel_file, mud_file, mlir_file, cali_table]:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  - {file} ({size:,} bytes)")
    
    print("\n🚀 Ready for MaixCam deployment!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
