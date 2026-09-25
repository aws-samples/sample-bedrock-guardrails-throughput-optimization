#!/usr/bin/env python3
"""
Teardown: delete the test guardrail created by setup_guardrail.py.

A guardrail is billed per text unit evaluated, so an idle one costs nothing,
but it counts against your account's guardrail quota. Delete it when you're done.

Usage:
    export AWS_PROFILE=your-profile
    python3 delete_guardrail.py --guardrail-id <id> [--region us-east-1]
"""
import argparse

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"


def main():
    parser = argparse.ArgumentParser(description="Delete a test guardrail")
    parser.add_argument("--guardrail-id", required=True, help="Guardrail ID to delete")
    parser.add_argument("--region", default=REGION, help=f"AWS region (default: {REGION})")
    args = parser.parse_args()

    client = boto3.client("bedrock", region_name=args.region)
    try:
        client.delete_guardrail(guardrailIdentifier=args.guardrail_id)
        print(f"Deleted guardrail {args.guardrail_id} in {args.region}.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ResourceNotFoundException":
            print(f"Guardrail {args.guardrail_id} not found (already deleted?).")
        else:
            raise


if __name__ == "__main__":
    main()
