import re
import json
import string
from pathlib import Path

import click
from tqdm import tqdm

from constants import (
    RESTAURANT, CATEGORIES, BUSINESS_ID, REVIEW_ID, TEXT, STARS, BAD_REVIEW_FILE, GOOD_REVIEW_FILE,
    MAX_LEN, ENCODING
)

RE_WHITESPACE = r"|".join([el + "+" for el in list(string.whitespace[1:])])


def get_total_rows(input_file):
    total_rows = sum(1 for line in input_file)
    input_file.seek(0)
    return total_rows


def get_business_id_by_label(path, label):
    business_ids = set()
    with open(path, "r", encoding=ENCODING) as jfile:
        total_rows = get_total_rows(jfile)

        for business in tqdm(jfile, total=total_rows):
            try:
                business = json.loads(business)
            except json.JSONDecodeError as ex:
                print("JSONDecodeError", ex)
                continue

            categories = business.get(CATEGORIES)
            business_id = business.get(BUSINESS_ID)
            if categories and business_id and (label in categories.lower()):
                business_ids.add(business_id)

    return business_ids


def save_reviews(review_path, output_dir, business_ids, max_len):
    printable_set = set(string.printable)

    with open(review_path, "r", encoding=ENCODING) as review_file,\
         open(output_dir / BAD_REVIEW_FILE, "w", encoding=ENCODING) as bad_review,\
         open(output_dir / GOOD_REVIEW_FILE, "w", encoding=ENCODING) as good_review:

        stars_to_file_map = {1: bad_review, 2: bad_review, 5: good_review}

        total_rows = get_total_rows(review_file)

        for review in tqdm(review_file, total=total_rows):
            try:
                review = json.loads(review)
            except json.JSONDecodeError as ex:
                print("JSONDecodeError", ex)
                continue

            review_id = review.get(REVIEW_ID)
            business_id = review.get(BUSINESS_ID)
            business_id_is_valid = business_id and (business_id in business_ids)
            stars = review.get(STARS)
            stars_is_valid = stars and stars.is_integer()
            text = review.get(TEXT)

            if business_id_is_valid and stars_is_valid and text and review_id:
                stars = int(stars)
                if stars in {1, 2, 5} and set(text).issubset(printable_set):
                    text = re.sub(r" +", " ", re.sub(RE_WHITESPACE, " ", text))
                    if len(text) <= max_len:
                        stars_to_file_map[stars].write(json.dumps({REVIEW_ID: review_id,
                                                                   TEXT: text}))
                        stars_to_file_map[stars].write("\n")


@click.command()
@click.option("--business-file", required=True, type=str)
@click.option("--reviews-file", required=True, type=str)
@click.option("--output-dir", required=True, type=str)
def main(business_file, reviews_file, output_dir):
    print(f"Get business ids with {RESTAURANT} in {CATEGORIES}:")
    business_ids = get_business_id_by_label(business_file, RESTAURANT)

    output_dir = Path(output_dir)
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    print(f"Save good and bad reviews into files:"
          f"\n- {output_dir / BAD_REVIEW_FILE}\n- {output_dir / GOOD_REVIEW_FILE}")
    save_reviews(reviews_file, output_dir, business_ids, MAX_LEN)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
