import boto3
import json
import csv
from decimal import Decimal

import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def decimal_to_float(obj):
    if isinstance(obj, list):
        result = []
        for i in obj:
            result.append(decimal_to_float(i))
        return result
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = decimal_to_float(v)
        return result
    elif isinstance(obj, Decimal):
        return float(obj)
    return obj

def main():
    session = boto3.Session(profile_name="mlops", region_name="ap-southeast-1")
    dynamodb = session.resource("dynamodb")  # resource, not client

    table = dynamodb.Table("SensorData")

    # Scan table (full pagination)
    items = []
    response = table.scan()
    items.extend(response.get('Items', []))

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response.get('Items', []))

    # Convert Decimals for serialization
    clean_items = decimal_to_float(items)
    
    if clean_items:
        keys = clean_items[0].keys()
        with open("sensor_data.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(clean_items)

if __name__=="__main__":
    main()