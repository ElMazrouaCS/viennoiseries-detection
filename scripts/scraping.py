from icrawler.builtin import BingImageCrawler
import os, shutil

# Classes et requêtes
classes_requetes = {
    "croissant": [
        "croissant boulangerie vitrine",
        "croissant french bakery plateau",
        "croissant café table",
        "croissant pâtisserie française"
    ],
    "chocolatine": [  # pain_au_chocolat renommé
        "pain au chocolat boulangerie",
        "chocolatine vitrine boulangerie",
        "pain chocolat panier osier",
        "pain au chocolat café"
    ],
    "eclair": [
        "éclair chocolat pâtisserie vitrine",
        "eclair café plateau",
        "éclair pâtisserie française",
        "eclair chocolat café"
    ],
    "macaron": [
        "macaron vitrine pâtisserie",
        "macaron plateau coloré",
        "macaron boîte ouverte",
        "macaron pâtisserie française"
    ],
    "tarte": [
        "tarte fraises boulangerie vitrine",
        "tarte fruits pâtisserie",
        "tarte framboise vitrine",
        "tarte citron meringuée boulangerie"
    ],
    "millefeuille": [
        "millefeuille pâtisserie française vitrine",
        "millefeuille tranche plateau",
        "millefeuille boulangerie",
        "mille feuille café table"
    ],
    "chausson": [
        "chausson aux pommes boulangerie",
        "chausson pommes corbeille",
        "chausson aux pommes vitrine",
        "chausson feuilleté boulangerie"
    ]
}

# Dossier final 
output_dir = "dataset/train/images"
os.makedirs(output_dir, exist_ok=True)

for classe, requetes in classes_requetes.items():
    print(f"\nScraping : {classe}")
    global_counter = 1  # compteur unique
    for i, q in enumerate(requetes):
        tmp_dir = f"dataset/tmp/{classe}/q{i}"
        os.makedirs(tmp_dir, exist_ok=True)
        
        crawler = BingImageCrawler(
            storage={"root_dir": tmp_dir},
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=4
        )
        crawler.crawl(keyword=q, max_num=30)
        
        # Déplacer vers le dossier final avec noms uniques
        for f in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, f)
            ext = os.path.splitext(f)[1]
            dst = os.path.join(output_dir, f"{classe}_{global_counter:06d}{ext}")
            shutil.move(src, dst)
            global_counter += 1
        
        shutil.rmtree(tmp_dir)  # clean temp

    print(f"{classe}: {global_counter - 1} images")

# Clean tmp global
shutil.rmtree("dataset/tmp", ignore_errors=True)

print("\nTotal par classe dans dataset/train/images :")
for classe in classes_requetes:
    n = len([f for f in os.listdir(output_dir) if f.startswith(classe)])
    print(f"  {classe}: {n} images")