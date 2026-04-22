#!/usr/bin/env python3
"""
Step 1: Create a test guardrail in your Isengard account.
Run this once, then use the guardrail ID in the test script.

Usage:
    export AWS_PROFILE=your-isengard-profile
    python3 setup_guardrail.py
"""

import boto3
import json

REGION = "us-east-1"

def create_test_guardrail():
    client = boto3.client("bedrock", region_name=REGION)

    response = client.create_guardrail(
        name="batch-test-guardrail",
        description="Test guardrail for validating content block batching behavior",
        contentPolicyConfig={
            "filtersConfig": [
                {
                    "type": "SEXUAL",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "VIOLENCE",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "HATE",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "INSULTS",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "MISCONDUCT",
                    "inputStrength": "HIGH",
                    "outputStrength": "HIGH"
                },
                {
                    "type": "PROMPT_ATTACK",
                    "inputStrength": "HIGH",
                    "outputStrength": "NONE"
                }
            ]
        },
        blockedInputMessaging="Input blocked by guardrail.",
        blockedOutputsMessaging="Output blocked by guardrail."
    )

    guardrail_id = response["guardrailId"]
    version = response["version"]

    print(f"Guardrail created:")
    print(f"  ID:      {guardrail_id}")
    print(f"  Version: {version}")
    print(f"  Region:  {REGION}")
    print(f"\nNow run: python3 test_batch.py --guardrail-id {guardrail_id}")

    return guardrail_id

if __name__ == "__main__":
    create_test_guardrail()
