import click
import os
import pathlib
import sys
#
from training import run_train


# Add the directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    
@click.command()
@click.argument('data_dir', type=click.Path(exists=True)) #Path to data directory
@click.option('--train_code', help='Train categories', default=True, is_flag=True)
def main_task(train_code, data_dir):
    if train_code:
        x = 'text'
        y = 'category'
        print('REPORT: \n {}'.format(run_train(data_dir, x, y)))
    else:
        click.UsageError('Illegal user: Please indicate a running option. ' \
                         'Type --help for more information of the available ' \
                         'options.')


if __name__ == '__main__':
    main_task()