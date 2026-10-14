from PIL import Image
import os

images_dir = "dataset/train/images"

print("Nettoyage des images corrompues ou trop petites...")

supprimees_total = 0
for f in os.listdir(images_dir):
    path = os.path.join(images_dir, f)
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)
        if img.size[0] < 100 or img.size[1] < 100:
            os.remove(path)
            supprimees_total += 1
    except:
        os.remove(path)
        supprimees_total += 1

print(f"Total supprimées : {supprimees_total}")

# Statistiques finales
classes = sorted(list(set(f.split('_')[0] for f in os.listdir(images_dir))))
print("\nImages restantes par classe :")
for classe in classes:
    n = len([f for f in os.listdir(images_dir) if f.startswith(classe)])
    print(f"  {classe}: {n}")