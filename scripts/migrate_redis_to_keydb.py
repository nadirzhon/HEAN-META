#!/usr/bin/env python3
"""
Migrate data from Redis to KeyDB.

This script safely migrates all data from an existing Redis instance
to a new KeyDB instance with progress tracking and error handling.

Usage:
    python migrate_redis_to_keydb.py --redis-host old-redis --keydb-host localhost

Requirements:
    pip install redis tqdm
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

try:
    import redis
    from tqdm import tqdm
except ImportError:
    print("Error: Missing dependencies. Install with: pip install redis tqdm")
    sys.exit(1)


def migrate_redis_to_keydb(
    redis_host: str = "old-redis",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str | None = None,
    keydb_host: str = "localhost",
    keydb_port: int = 6379,
    keydb_db: int = 0,
    keydb_password: str | None = None,
    batch_size: int = 1000,
) -> Tuple[int, List[Tuple[bytes, str]]]:
    """
    Safely migrate data from Redis to KeyDB.

    Args:
        redis_host: Redis source hostname
        redis_port: Redis source port
        redis_db: Redis source database number
        redis_password: Redis source password (if any)
        keydb_host: KeyDB destination hostname
        keydb_port: KeyDB destination port
        keydb_db: KeyDB destination database number
        keydb_password: KeyDB destination password (if any)
        batch_size: Number of keys to process in each batch

    Returns:
        Tuple of (migrated_count, errors_list)
    """
    print("🔄 Starting Redis to KeyDB migration...")

    # Connect to Redis (source)
    try:
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # Keep as bytes
        )
        redis_client.ping()
        print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        sys.exit(1)

    # Connect to KeyDB (destination)
    try:
        keydb_client = redis.Redis(
            host=keydb_host,
            port=keydb_port,
            db=keydb_db,
            password=keydb_password,
            decode_responses=False,  # Keep as bytes
        )
        keydb_client.ping()
        print(f"✅ Connected to KeyDB at {keydb_host}:{keydb_port}")
    except Exception as e:
        print(f"❌ Failed to connect to KeyDB: {e}")
        sys.exit(1)

    # Get all keys from Redis
    print("📊 Scanning keys from Redis...")
    try:
        keys = redis_client.keys("*")
        total_keys = len(keys)
        print(f"📦 Found {total_keys} keys to migrate")
    except Exception as e:
        print(f"❌ Failed to scan keys: {e}")
        sys.exit(1)

    if total_keys == 0:
        print("ℹ️  No keys to migrate")
        return 0, []

    # Migrate keys with progress bar
    migrated = 0
    errors: List[Tuple[bytes, str]] = []

    with tqdm(total=total_keys, desc="Migrating", unit="keys") as pbar:
        for key in keys:
            try:
                # Get key type
                key_type = redis_client.type(key)
                ttl = redis_client.ttl(key)

                # Migrate based on type
                if key_type == b"string":
                    value = redis_client.get(key)
                    keydb_client.set(key, value)

                elif key_type == b"hash":
                    value = redis_client.hgetall(key)
                    if value:
                        keydb_client.hset(key, mapping=value)

                elif key_type == b"list":
                    value = redis_client.lrange(key, 0, -1)
                    if value:
                        keydb_client.rpush(key, *value)

                elif key_type == b"set":
                    value = redis_client.smembers(key)
                    if value:
                        keydb_client.sadd(key, *value)

                elif key_type == b"zset":
                    value = redis_client.zrange(key, 0, -1, withscores=True)
                    if value:
                        # Convert list of (member, score) tuples to dict
                        score_dict = {member: score for member, score in value}
                        keydb_client.zadd(key, score_dict)

                else:
                    errors.append((key, f"Unknown type: {key_type}"))
                    pbar.update(1)
                    continue

                # Restore TTL if exists
                if ttl > 0:
                    keydb_client.expire(key, ttl)

                migrated += 1
                pbar.update(1)

            except Exception as e:
                errors.append((key, str(e)))
                pbar.update(1)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 Migration Summary")
    print("=" * 60)
    print(f"Total keys:     {total_keys}")
    print(f"✅ Migrated:     {migrated}")
    print(f"❌ Errors:       {len(errors)}")
    print(f"Success rate:   {(migrated/total_keys*100):.2f}%")

    if errors:
        print("\n⚠️  Errors encountered:")
        for key, error in errors[:10]:  # Show first 10 errors
            key_str = key.decode("utf-8", errors="replace")
            print(f"  - {key_str}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    return migrated, errors


def verify_migration(
    redis_host: str = "old-redis",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: str | None = None,
    keydb_host: str = "localhost",
    keydb_port: int = 6379,
    keydb_db: int = 0,
    keydb_password: str | None = None,
    sample_size: int = 100,
) -> Dict[str, int]:
    """
    Verify migration integrity by comparing a sample of keys.

    Returns:
        Dictionary with verification statistics
    """
    print("\n🔍 Verifying migration integrity...")

    # Connect to both
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=False,
    )
    keydb_client = redis.Redis(
        host=keydb_host,
        port=keydb_port,
        db=keydb_db,
        password=keydb_password,
        decode_responses=False,
    )

    redis_keys = set(redis_client.keys("*"))
    keydb_keys = set(keydb_client.keys("*"))

    missing = redis_keys - keydb_keys
    extra = keydb_keys - redis_keys

    print(f"📊 Redis keys:  {len(redis_keys)}")
    print(f"📊 KeyDB keys:  {len(keydb_keys)}")
    print(f"⚠️  Missing:    {len(missing)}")
    print(f"ℹ️  Extra:      {len(extra)}")

    # Sample check
    sample_keys = list(redis_keys)[: min(sample_size, len(redis_keys))]
    mismatches = 0

    print(f"\n🔬 Checking {len(sample_keys)} sample keys for data integrity...")

    for key in tqdm(sample_keys, desc="Verifying", unit="keys"):
        try:
            redis_val = redis_client.get(key)
            keydb_val = keydb_client.get(key)
            if redis_val != keydb_val:
                mismatches += 1
                key_str = key.decode("utf-8", errors="replace")
                print(f"⚠️  Mismatch: {key_str}")
        except Exception as e:
            print(f"❌ Error checking key: {e}")
            mismatches += 1

    matches = len(sample_keys) - mismatches
    print(f"\n✅ Sample check: {matches}/{len(sample_keys)} keys match")

    return {
        "redis_keys": len(redis_keys),
        "keydb_keys": len(keydb_keys),
        "missing": len(missing),
        "extra": len(extra),
        "sample_size": len(sample_keys),
        "matches": matches,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate data from Redis to KeyDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate from local Redis to local KeyDB
  python migrate_redis_to_keydb.py

  # Migrate from remote Redis to local KeyDB
  python migrate_redis_to_keydb.py --redis-host 192.168.1.100 --keydb-host localhost

  # Migrate with passwords
  python migrate_redis_to_keydb.py --redis-password secret1 --keydb-password secret2

  # Skip verification
  python migrate_redis_to_keydb.py --no-verify
        """,
    )

    parser.add_argument(
        "--redis-host", default="redis", help="Redis source hostname (default: redis)"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis source port (default: 6379)"
    )
    parser.add_argument(
        "--redis-db", type=int, default=0, help="Redis source database (default: 0)"
    )
    parser.add_argument("--redis-password", help="Redis source password")
    parser.add_argument(
        "--keydb-host",
        default="localhost",
        help="KeyDB destination hostname (default: localhost)",
    )
    parser.add_argument(
        "--keydb-port",
        type=int,
        default=6379,
        help="KeyDB destination port (default: 6379)",
    )
    parser.add_argument(
        "--keydb-db", type=int, default=0, help="KeyDB destination database (default: 0)"
    )
    parser.add_argument("--keydb-password", help="KeyDB destination password")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for migration (default: 1000)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Sample size for verification (default: 100)",
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip verification step"
    )

    args = parser.parse_args()

    # Run migration
    start_time = time.time()
    migrated, errors = migrate_redis_to_keydb(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_password=args.redis_password,
        keydb_host=args.keydb_host,
        keydb_port=args.keydb_port,
        keydb_db=args.keydb_db,
        keydb_password=args.keydb_password,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start_time

    print(f"\n⏱️  Migration completed in {elapsed:.2f} seconds")
    print(f"📈 Throughput: {migrated/elapsed:.0f} keys/sec")

    # Run verification (unless disabled)
    if not args.no_verify:
        verify_migration(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
            redis_password=args.redis_password,
            keydb_host=args.keydb_host,
            keydb_port=args.keydb_port,
            keydb_db=args.keydb_db,
            keydb_password=args.keydb_password,
            sample_size=args.sample_size,
        )

    # Exit with error if there were migration failures
    if errors:
        print("\n⚠️  Migration completed with errors. Please review.")
        sys.exit(1)
    else:
        print("\n✅ Migration completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
