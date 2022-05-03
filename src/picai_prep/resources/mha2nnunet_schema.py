mha2nnunet_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "archive": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "description": "Custom patient ID. If not provided, uses the 'PatientID' metadata",
                        "type": "string"
                    },
                    "study_id": {
                        "description": "Custom study ID. If not provided, uses the 'StudyInstanceUID' metadata",
                        "type": "string"
                    },
                    "scan_paths": {
                        "description": "Paths to mha collection. Can be relative from --input or absolute",
                        "type": "array",
                        "items": {
                            "type": "string",
                            "uniqueItems": True
                        }
                    },
                    "annotation_path": {
                        "description": "Path to annotation. Can be relative from --input or absolute",
                        "type": "string"
                    }
                },
                "required": [
                    "scan_paths"
                ],
                "additionalProperties": False
            }
        },
        "preprocessing": {
            "type": "object",
            "description": "Preprocessing parameters",
            "properties": {
                "align_physical_space": {
                    "description": "...",
                    "type": "boolean"
                },
                "crop_to_first_physical_centre": {
                    "description": "...",
                    "type": "boolean"
                },
                "physical_size": {
                    "description": "...",
                    "$ref": "#/$defs/3d"
                },
                "matrix_size": {
                    "description": "Defaults to [20, 160, 160] if neither this or 'physical_size' is set.",
                    "$ref": "#/$defs/3d"
                },
                "spacing": {
                    "description": "...",
                    "$ref": "#/$defs/3d"
                }
            },
            "additionalProperties": False
        },
        "dataset_json": {
            "type": "object",
            "description": "nnUnet requires a dataset.json file",
            "properties": {
                "task": {
                    "type": "string"
                },
                "name": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                },
                "tensorImageSize": {
                    "type": "string",
                    "description": "3D or 4D, defaults to 4D",
                    "pattern": "^\\d[Dd]$"
                },
                "reference": {
                    "type": "string",
                    "description": "website of the dataset, if available"
                },
                "licence": {
                    "type": "string"
                },
                "release": {
                    "type": "string"
                },
                "modality": {
                    "$ref": "#/$defs/1a2a3a",
                    "description": "modality names. must be in the same order as the images"
                },
                "labels": {
                    "$ref": "#/$defs/1a2a3a",
                    "description": "..."
                }
            },
            "additionalProperties": True,
            "required": [
                "modality",
                "labels"
            ]
        }
    },
    "additionalProperties": False,
    "required": [
        "archive",
        "preprocessing",
        "dataset_json"
    ],
    "$defs": {
        "3d": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "number"
            }
        },
        "1a2a3a": {
            "type": "object",
            "patternProperties": {
                "^\\d+$": {
                    "type": "string"
                }
            }
        }
    }
}
