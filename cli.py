import click
from bot.client import get_client
from bot.orders import place_order
from bot.validators import *
from bot.logging_config import setup_logger
import os



setup_logger()

@click.command()
@click.option('--symbol', required=True, help="e.g. BTCUSDT")
@click.option('--side', required=True, help="BUY or SELL")
@click.option('--type', 'order_type', required=True, help="MARKET or LIMIT")
@click.option('--quantity', required=True, type=float)
@click.option('--price', type=float, default=None)

def main(symbol, side, order_type, quantity, price):
    try:
        validate_side(side)
        validate_order_type(order_type)
        validate_quantity(quantity)
        validate_price(price, order_type)

        client = get_client()

        print("\n📌 Order Summary")
        print(f"Symbol: {symbol}")
        print(f"Side: {side}")
        print(f"Type: {order_type}")
        print(f"Quantity: {quantity}")
        print(f"Price: {price}")

        order = place_order(client, symbol, side, order_type, quantity, price)

        print("\n✅ SUCCESS")
        print(f"Order ID: {order.get('orderId')}")
        print(f"Status: {order.get('status')}")
        print(f"Executed Qty: {order.get('executedQty')}")
        print(f"Avg Price: {order.get('avgPrice')}")

    except Exception as e:
        print("\n❌ ERROR:", str(e))

if os.getenv("MOCK_MODE") == "True":
    print("\n⚠ Running in MOCK MODE (no real orders)")

if __name__ == "__main__":
    main()