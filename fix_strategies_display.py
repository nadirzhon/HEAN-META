#!/usr/bin/env python3
"""Fix strategies display issue."""

import httpx
import asyncio

API_BASE = "http://localhost:8000"

async def main():
    async with httpx.AsyncClient() as client:
        # 1. Reset state
        print("1. Resetting paper state...")
        await client.post(f"{API_BASE}/orders/paper/reset_state")
        
        # 2. Reset killswitch
        print("2. Resetting killswitch...")
        await client.post(f"{API_BASE}/risk/reset_killswitch")
        
        # 3. Restart engine
        print("3. Restarting engine...")
        await client.post(f"{API_BASE}/engine/restart")
        await asyncio.sleep(5)
        
        # 4. Check strategies
        print("4. Checking strategies...")
        strategies = await client.get(f"{API_BASE}/strategies")
        print(f"Strategies: {strategies.json()}")
        
        # 5. Check risk status
        print("5. Checking risk status...")
        risk = await client.get(f"{API_BASE}/risk/status")
        print(f"Risk: {risk.json()}")
        
        # 6. Place test order
        print("6. Placing test order...")
        order = await client.post(
            f"{API_BASE}/orders/test",
            json={"symbol": "BTCUSDT", "side": "buy", "size": 0.001, "price": 50000}
        )
        print(f"Order: {order.json()}")
        await asyncio.sleep(2)
        
        # 7. Check orders
        print("7. Checking orders...")
        orders = await client.get(f"{API_BASE}/orders")
        print(f"Orders count: {len(orders.json())}")
        
        # 8. Check positions
        print("8. Checking positions...")
        positions = await client.get(f"{API_BASE}/orders/positions")
        print(f"Positions count: {len(positions.json())}")

if __name__ == "__main__":
    asyncio.run(main())
