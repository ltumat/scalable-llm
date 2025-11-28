def main():
    from finetune import get_train_data, FinetuneConfig

    ds = get_train_data(FinetuneConfig())

    print(ds[0])


if __name__ == "__main__":
    main()
