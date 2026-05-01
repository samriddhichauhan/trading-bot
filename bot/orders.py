import logging
import os
import random
import time
import uuid

def place_order(client, symbol, side, order_type, quantity, price=None):
    request_id = str(uuid.uuid4())[:8]  # short unique ID
    start_time = time.time()

    try:
        mock_mode = os.getenv("MOCK_MODE", "False") == "True"

        logging.info(f"[{request_id}] START Order")
        logging.info(f"[{request_id}] Input -> symbol={symbol}, side={side}, type={order_type}, qty={quantity}, price={price}")
        logging.info(f"[{request_id}] Mode -> {'MOCK' if mock_mode else 'REAL'}")

        if mock_mode:
            time.sleep(1)

            fake_price = price if price else round(random.uniform(50000, 70000), 2)

            response = {
                "orderId": random.randint(100000, 999999),
                "status": "FILLED" if order_type == "MARKET" else "NEW",
                "executedQty": quantity if order_type == "MARKET" else 0,
                "avgPrice": fake_price
            }

        else:
            if order_type == "MARKET":
                response = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=quantity
                )

            elif order_type == "LIMIT":
                response = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    quantity=quantity,
                    price=price,
                    timeInForce="GTC"
                )

        end_time = time.time()
        execution_time = round(end_time - start_time, 3)

        logging.info(f"[{request_id}] Response -> {response}")
        logging.info(f"[{request_id}] Execution Time -> {execution_time}s")
        logging.info(f"[{request_id}] END Order\n")

        return response

    except Exception as e:
        logging.error(f"[{request_id}] ERROR -> {str(e)}", exc_info=True)
        raise