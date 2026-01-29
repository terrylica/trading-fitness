# Data Infrastructure

> Bigblack is the sole data storage for range bars and tick cache.

**← [Back to trading-fitness](../../CLAUDE.md)**

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────────────────┐
│   Local (macOS)     │         │   Bigblack (Linux GPU Workstation)  │
│                     │  SSH    │                                     │
│  Code + Analysis    │ ──────▶ │  ClickHouse: rangebar_cache         │
│  (no data caching)  │         │  Tick Cache: ~/.cache/rangebar/     │
└─────────────────────┘         └─────────────────────────────────────┘
```

## Why Bigblack?

| Concern       | Local (macOS)       | Bigblack (Linux)      |
| ------------- | ------------------- | --------------------- |
| Storage       | Limited (256GB SSD) | Abundant (2TB+ NVMe)  |
| CPU for fetch | 14 cores            | 32 cores              |
| ClickHouse    | mise-installed      | Native installation   |
| Network       | Home connection     | Data center proximity |

## Components

### ClickHouse Database

- **Host**: `bigblack` (via SSH or ZeroTier: 172.25.236.1)
- **Database**: `rangebar_cache`
- **Table**: `range_bars`
- **Engine**: `ReplacingMergeTree(computed_at)`

```sql
-- Check data status
SELECT symbol, threshold_decimal_bps, count(),
       min(toDate(timestamp_ms/1000)), max(toDate(timestamp_ms/1000))
FROM rangebar_cache.range_bars
GROUP BY 1,2 ORDER BY 1,2;
```

### Tick Cache (Parquet)

- **Location**: `bigblack:~/.cache/rangebar/ticks/`
- **Format**: Parquet files per symbol/month
- **Structure**: `BINANCE_SPOT_{SYMBOL}/{YYYY-MM}.parquet`

## Data Management

### Precompute Range Bars

```bash
# On bigblack (recommended - 4 workers = 1 per symbol to avoid tick cache race)
ssh bigblack "cd ~/eon/trading-fitness/packages/ith-python && \
  nohup ~/.local/bin/uv run python ../../scripts/precompute_historical_parallel.py --workers 4 \
  > ~/eon/trading-fitness/logs/precompute_\$(date +%Y%m%d).log 2>&1 &"

# Monitor progress
ssh bigblack "tail -f ~/eon/trading-fitness/logs/precompute_*.log"

# Check tick cache growth (first-tier caching)
ssh bigblack "du -sh ~/.cache/rangebar/ticks/"
```

**Parallelization Strategy**: Max 4 workers (one per symbol) because thresholds for the same symbol share tick cache files. Running multiple thresholds for the same symbol in parallel causes Parquet race conditions.

### Sync Code to Bigblack

```bash
# From local project root
rsync -avz \
  --exclude='.venv' \
  --exclude='artifacts' \
  --exclude='logs' \
  --exclude='.git' \
  --exclude='target' \
  --exclude='__pycache__' \
  . bigblack:~/eon/trading-fitness/
```

### Query Data Remotely

```bash
# Check total bars
ssh bigblack "clickhouse-client --query 'SELECT count() FROM rangebar_cache.range_bars'"

# Export to local (if needed for specific analysis)
ssh bigblack "clickhouse-client --query 'SELECT * FROM rangebar_cache.range_bars WHERE symbol=\"BTCUSDT\" FORMAT Parquet'" > btcusdt.parquet
```

## Corruption Recovery

If Parquet tick cache becomes corrupted:

```bash
# Clear corrupted cache (safe - ClickHouse data intact)
ssh bigblack "rm -rf ~/.cache/rangebar/ticks/*"

# Re-run precompute (will refetch from Binance)
ssh bigblack "cd ~/eon/trading-fitness/packages/ith-python && \
  ~/.local/bin/uv run python ../../scripts/precompute_historical_parallel.py --workers 8"
```

## Timestamp Precision by Year

See [CLAUDE.md](../../CLAUDE.md#timestamp-precision-by-year-gotcha) for the timestamp precision gotcha (2022-2023 ms vs 2024+ ns).

---

## Data Coverage

Updated: 2026-01-26

| Symbol  | 25dbps | 50dbps | 100dbps | 250dbps |
| ------- | ------ | ------ | ------- | ------- |
| BTCUSDT | 0.7y   | 4.1y   | 4.1y    | 3.9y    |
| ETHUSDT | 4.1y   | 0.5y   | 0.1y    | 0.6y    |
| SOLUSDT | 4.1y   | 1.8y   | 2.2y    | 4.1y    |
| BNBUSDT | -      | 4.1y   | -       | 4.1y    |

**Target**: All combinations with 4+ years coverage.

## Local Development (No Data)

For local development without bigblack:

1. Use small synthetic datasets for unit tests
2. Mock ClickHouse responses for integration tests
3. Run full pipeline only on bigblack

```python
# In tests, use fixtures instead of real data
@pytest.fixture
def sample_nav():
    return np.random.randn(1000).cumsum() + 100
```
