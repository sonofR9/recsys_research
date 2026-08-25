from data import DatasetManager

if __name__ == "__main__":
    manager = DatasetManager(
        main_parquet="data/main.parquet",
        embeddings_parquet="data/embeddings.parquet",
        data_dir="data/processed",
        counter_columns=["user_likes_count", "item_popularity"],
    )

    print(f"Available days: {manager.get_available_days()}")

    dataset = manager.create_dataset(days=0)
    print(f"Dataset size: {len(dataset)}")

    dataloader = manager.create_dataloader(dataset, batch_size=256)

    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch size: {batch['categorical_features']['item_id'].shape[0]}")
        break

    print("\nDay-by-day training example:")
    for day in manager.get_available_days()[:3]:
        dataset = manager.create_dataset(days=day)
        dataloader = manager.create_dataloader(dataset, shuffle=False)
        print(f"Day {day}: {len(dataset)} samples")

    print("\nMultiple days example:")
    dataset = manager.create_dataset(days=[0, 1, 2])
    print(f"Days 0-2 combined: {len(dataset)} samples")
