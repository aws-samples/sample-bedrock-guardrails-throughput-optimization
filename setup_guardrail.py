#!/usr/bin/env python3
"""
Create the test guardrail that test_batch.py runs against.
Run this once, then pass the printed guardrail ID to the test script.

Usage:
    export AWS_PROFILE=your-profile
    python3 setup_guardrail.py [--region us-east-1]
"""

import argparse

import boto3

REGION = "us-east-1"


def create_test_guardrail(region):
    client = boto3.client("bedrock", region_name=region)

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

    print("Guardrail created:")
    print(f"  ID:      {guardrail_id}")
    print(f"  Version: {version}")
    print(f"  Region:  {region}")
    print(f"\nNow run: python3 test_batch.py --guardrail-id {guardrail_id} --region {region}")

    return guardrail_id


def main():
    parser = argparse.ArgumentParser(description="Create a test guardrail")
    parser.add_argument("--region", default=REGION, help=f"AWS region (default: {REGION})")
    args = parser.parse_args()
    create_test_guardrail(args.region)


if __name__ == "__main__":
    main()
