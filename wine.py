import pandas as pd
import numpy as np
import click
import json
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s %(filename)s:%(lineno)s:: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--source')
@click.option('--target')
def prepare(source, target):
    output = open(target, "w")
    blog_ids = set()
    blogs = []
    with open(source) as f:
        for line in f:
            data = json.loads(line)['data']
            logger.info(f"card list info {len(data['cardlistInfo'])}, cards {len(data['cards'])}")
            for card in data['cards']:
                if 'card_group' not in card:
                    continue
                if len(card['card_group']) == 0:
                    continue
                if 'mblog' not in card['card_group'][0]:
                    continue
                mblog = card['card_group'][0]['mblog']
                blog_id = mblog['id']
                ad_marked = mblog['ad_marked']
                created_at = mblog['created_at']
                city = mblog.get('status_city')
                country = mblog.get('status_country')
                province = mblog.get('status_province')
                text = mblog.get('text')
                user = mblog['user']
                user_screen_name = user['screen_name']
                user_gender = user['gender']
                user_description = user['description']
                #logger.info(f"blog id {blog_id}, created_at {created_at}, ad_marked {ad_marked}, country {country}, province {province}, city {city}, text {text}")
                if blog_id in blog_ids:
                    continue
                blog_ids.add(blog_id)
                blogs.append({
                    'id': blog_id,
                    'ad_marked': ad_marked,
                    'created_at': created_at,
                    'city': city,
                    'country': country,
                    'province': province,
                    'text': text,
                    'user_screen_name': user_screen_name,
                    'user_gender': user_gender,
                    'user_description': user_description,
                })

    for blog in blogs:
        output.write(json.dumps(blog))
        output.write('\n')
    output.close()
    logger.info(f"process {len(blogs)} blogs")



if __name__ == '__main__':
    cli()
