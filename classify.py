import click
import logging
import os
import pathlib
import sys
#
from predicting import run_predict
#from preprocessing import clean_text


# Add the directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    
                    
@click.command()
@click.option('--infer_code', help='Infer category', default=True, is_flag=True)
@click.argument('model', type=click.Path(exists=True)) #Path to data directory
@click.option('--files', nargs=0, required=True)
@click.argument('files', nargs=-1)
# bonus track?
@click.option('--pair_code', help='Infer 2 categories', default=False, is_flag=True)
def main_task(infer_code, model, files, pair_code):
    if infer_code:
        for f in files:
            prediction = run_predict(model, f)
            print('{} {} {}'.format(prediction[0], prediction[1], "2nd option: {}".format(prediction[3]) if pair_code else ""))
    else:
        click.UsageError('Illegal user: Please indicate a running option. ' \
                         'Type --help for more information of the available ' \
                         'options.')


if __name__ == '__main__':
        main_task()

