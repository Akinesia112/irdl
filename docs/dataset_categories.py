"""Single source of truth for dataset grouping in the docs."""

DATASET_CATEGORIES = [
    {
        "slug": "room_impulse_responses",
        "title": "Room impulse responses",
        "datasets": [
            {
                "name": "irdl.MiracleDataset",
                "description_rst": None,
            },
            {
                "name": "irdl.SrirachaDataset",
                "description_rst": None,
            },
        ],
    },
    {
        "slug": "head_related_impulse_responses",
        "title": "Head-related impulse responses",
        "datasets": [
            {
                "name": "irdl.FabianDataset",
                "description_rst": None,
            },
        ],
    },
]
