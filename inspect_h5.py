import h5py

with h5py.File("data/raw/METR-LA.h5", "r") as f:
    print("Top-level keys:")
    for key in f.keys():
        print(" -", key)

        if isinstance(f[key], h5py.Group):
            print("   Sub-keys:")
            for sub in f[key].keys():
                print("     >", sub)
