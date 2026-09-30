def print_validation(
    customer_df,
    sample,
    topic_df,
):
    """
    Print final discovery statistics.
    """
    print("\n" + "=" * 70)
    print("DISCOVERY VALIDATION")
    print("=" * 70)

    print(
        f"\nUsable unique customer messages: "
        f"{len(customer_df):,}"
    )

    print(
        f"Discovery sample: "
        f"{len(sample):,}"
    )

    print(
        f"Candidate topics: "
        f"{len(topic_df):,}"
    )

    print(
        "\nSample coverage: "
        f"{len(sample) / len(customer_df):.2%}"
    )

    print(
        "\nTopic shares:"
    )

    for _, row in topic_df.iterrows():
        print(
            f"  {row['topic']}: "
            f"{row['topic_share']:.2%}"
        )