from pathlib import Path

# ---- Adjust ROOT if needed ----
ROOT = Path("/home/rr0110@DS.UAH.edu/quantum/medical/physionet.org/files/hmc-sleep-staging/1.1/recordings")

def main():
    print("[INFO] Current working directory:", Path.cwd())
    print("[INFO] ROOT set to:", ROOT)

    if ROOT.exists() and ROOT.is_dir():
        print("[OK] Found dataset directory:", ROOT)
        print("[INFO] Listing first 20 items:")
        for i, p in enumerate(sorted(ROOT.iterdir())):
            print(" ", p.name)
            if i >= 19:
                print(" ... (more files not shown)")
                break
    else:
        print("[ERROR] ROOT not found:", ROOT)
        parent = ROOT.parent
        if parent.exists():
            print("[HINT] Parent folder exists:", parent)
            print("       Contents of parent:")
            for p in sorted(parent.iterdir()):
                print("  -", p.name)
        else:
            print("[HINT] Parent folder also does not exist:", parent)

if __name__ == "__main__":
    main()
