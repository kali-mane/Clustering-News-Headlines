# main.py
from clustering import scrapping, preprocessing, k_means, hierarchical


def main():
    print("Starting run")
    scrapping.main()
    preprocessing.main()
    k_means.main()
    hierarchical.main()
    print("Run complete")


if __name__ == '__main__':
    main()
