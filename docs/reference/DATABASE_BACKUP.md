# Database Backup Strategy

## Overview

TimescaleDB backups are critical for data preservation. This document outlines backup strategies for Nova Aetus.

## Backup Methods

### 1. Automated pg_dump Backups (Recommended)

#### Daily Backups
```bash
#!/bin/bash
# daily_backup.sh

BACKUP_DIR="/var/backups/nova_aetus"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/nova_aetus_$DATE.sql"

mkdir -p $BACKUP_DIR

# Create backup
docker exec nova_aetus_db pg_dump -U postgres nova_aetus > $BACKUP_FILE

# Compress
gzip $BACKUP_FILE

# Keep only last 30 days
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

#### Weekly Full Backups + Daily Incremental
```bash
# Full backup (weekly)
pg_dump -Fc -U postgres nova_aetus > backup_full_$(date +%Y%W).dump

# Incremental backup (daily)
pg_basebackup -D /backup/incremental -Ft -z -P
```

### 2. TimescaleDB Continuous Aggregates Backup

For time-series data, consider backing up continuous aggregates separately:

```bash
# Backup hypertables
pg_dump -t market_bars -U postgres nova_aetus > market_bars_backup.sql

# Backup continuous aggregates (if using)
pg_dump -t <aggregate_table> -U postgres nova_aetus > aggregates_backup.sql
```

### 3. Docker Volume Backup

```bash
# Backup entire TimescaleDB volume
docker run --rm \
  -v nova_aetus_timescaledb_data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/timescaledb_volume_$(date +%Y%m%d).tar.gz /data
```

## Restore Procedures

### Restore from pg_dump
```bash
# Drop existing database (CAUTION: Destructive!)
docker exec -i nova_aetus_db psql -U postgres -c "DROP DATABASE nova_aetus;"
docker exec -i nova_aetus_db psql -U postgres -c "CREATE DATABASE nova_aetus;"

# Restore
gunzip -c backup_file.sql.gz | docker exec -i nova_aetus_db psql -U postgres nova_aetus
```

### Restore from volume backup
```bash
docker run --rm \
  -v nova_aetus_timescaledb_data:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/timescaledb_volume_YYYYMMDD.tar.gz -C /
```

## Backup Schedule Recommendations

- **Daily**: Incremental backups at 2 AM
- **Weekly**: Full backup on Sundays at 1 AM
- **Monthly**: Archive full backup to remote storage
- **Before Major Updates**: Manual backup

## Automation

### Cron Job Setup
```bash
# Add to crontab: crontab -e
0 2 * * * /path/to/daily_backup.sh
0 1 * * 0 /path/to/weekly_backup.sh
```

### systemd Timer (Alternative)
Create `/etc/systemd/system/nova-backup.timer`:
```ini
[Unit]
Description=Daily Nova Aetus Backup

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

## Remote Backup Storage

Consider storing backups remotely:
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- SFTP server

Example S3 upload:
```bash
aws s3 cp backup_file.sql.gz s3://nova-aetus-backups/daily/
```

## Verification

Always verify backups:
```bash
# Test restore on temporary database
docker exec -i nova_aetus_db createdb -U postgres test_restore
gunzip -c backup_file.sql.gz | docker exec -i nova_aetus_db psql -U postgres test_restore
docker exec -i nova_aetus_db psql -U postgres -d test_restore -c "SELECT COUNT(*) FROM market_bars;"
docker exec -i nova_aetus_db dropdb -U postgres test_restore
```

## Monitoring

Monitor backup success/failure:
- Check backup file sizes
- Verify backup timestamps
- Alert on backup failures (via Discord webhook)

## Retention Policy

- Daily backups: 30 days
- Weekly backups: 12 weeks (3 months)
- Monthly backups: 12 months (1 year)
