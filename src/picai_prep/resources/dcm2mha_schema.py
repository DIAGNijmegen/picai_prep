dcm2mha_schema = {
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
                    "path": {
                        "description": "Path to study directory. Can be relative from --input or absolute",
                        "type": "string"
                    }
                },
                "required": [
                    "path"
                ],
                "additionalProperties": False
            }
        },
        "mappings": {
            "type": "object",
            "description": "Filter the archive using a key/value system, each study can have at most 1 mapping",
            "patternProperties": {
                "^[A-Za-z_][A-Za-z0-9_]*$": {
                    "description": "Keys can be any of the items found in metadata.json, values can be any string",
                    "type": "object"
                }
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False,
    "required": [
        "archive",
        "mappings"
    ]
}
