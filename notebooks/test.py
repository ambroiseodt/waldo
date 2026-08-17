import seaborn as sns

pal = sns.light_palette("#5b6da9", n_colors=6, reverse=False)  # sns.cubehelix_palette(rot=0.2)
print(pal.as_hex()[:])
