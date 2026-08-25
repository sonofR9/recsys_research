#!/usr/bin/env python3
"""
Script to create a test_users.parquet file from sequential data.

This script extracts a sample of users from the test set to use for
candidate generation. It can be used to create the test_users.parquet
file required by generate_candidates.py.

Usage:
    python create_test_users.py --data_dir ../../data/ --size 50m --interaction likes --num_users 1000
"""

import click
import polars as pl
import pathlib as Path


@click.command()
@click.option('--data_dir', required=True, type=str, default='../../data/', show_default=True, help='Path to data directory')
@click.option(
    '--size',
    required=True,
    type=click.Choice(['50m', '500m', '5b']),
    default='50m',
    show_default=True,
    help='Dataset size'
)
@click.option(
    '--interaction',
    required=True,
    type=click.Choice(['likes', 'listens']),
    default='likes',
    show_default=True,
    help='Interaction type'
)
@click.option('--num_users', required=False, type=int, default=1000, show_default=True, help='Number of test users to sample')
@click.option('--output_path', required=True, type=str, default='./test_users.parquet', show_default=True, help='Output path for test_users.parquet')
@click.option('--seed', required=False, type=int, default=42, show_default=True, help='Random seed for sampling')
def main(
    data_dir: str,
    size: str,
    interaction: str,
    num_users: int,
    output_path: str,
    seed: int,
):
    """
    Create a test_users.parquet file by sampling users from the sequential data.
    
    The output file will contain a 'uid' column with user IDs that can be used
    for candidate generation.
    """
    print(f'Loading data from {data_dir}/sequential/{size}/{interaction}.parquet...')
    
    path = Path.Path(data_dir) / 'sequential' / size / interaction
    df = pl.scan_parquet(path.with_suffix('.parquet'))
    
    # Get all unique user IDs
    print('Extracting user IDs...')
    all_users = df.select('uid').unique().collect()
    
    total_users = len(all_users)
    print(f'Found {total_users} total users in the dataset')
    
    # Sample users
    if num_users > total_users:
        print(f'Warning: Requested {num_users} users but only {total_users} available. Using all users.')
        num_users = total_users
    
    print(f'Sampling {num_users} users with seed {seed}...')
    test_users = all_users.sample(n=num_users, seed=seed)
    
    # Save to parquet
    print(f'Saving test users to {output_path}...')
    test_users.write_parquet(output_path)
    
    print(f'\n{"="*60}')
    print(f'Successfully created test_users.parquet')
    print(f'{"="*60}')
    print(f'Total users in dataset: {total_users}')
    print(f'Sampled test users: {len(test_users)}')
    print(f'Output file: {output_path}')
    print(f'{"="*60}\n')
    
    # Show sample of the data
    print('Sample of test_users.parquet:')
    print(test_users.head(10))


if __name__ == '__main__':
    main()