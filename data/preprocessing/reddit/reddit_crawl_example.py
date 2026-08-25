import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.parse import (
    urlparse,
    urlunparse,
)

from datasets import (
    DatasetDict,
    Features,
    Value,
    load_from_disk,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from playwright.sync_api import sync_playwright

# This crawler has been written with AI support


DATASET_NAMES = [
    "profile_based",
    "in_domain",
    "cross_domain",
    "mixed",
]

POST_SEPARATOR = "<POST>"

# These are Reddit interface labels, not comment bodies. Cache entries
# containing these exact strings are treated as failures and retried.
INVALID_UI_TEXTS = {
    "Nutzermenü ausklappen",
    "Weitere Antworten",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Reddit comment text with Playwright and fill "
            "text_a/text_b in saved Hugging Face datasets."
        )
    )

    parser.add_argument(
        "--input-root",
        default="build_reddit_data/dataset_with_permalinks",
        help="Directory containing the permalink datasets.",
    )

    parser.add_argument(
        "--output-root",
        default="build_reddit_data/dataset_with_text",
        help="Directory in which translated datasets will be saved.",
    )

    parser.add_argument(
        "--cache-file",
        default=(
            "build_reddit_data/dataset_with_text/"
            "translation_cache.json"
        ),
        help="JSON checkpoint containing processed permalinks.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "Maximum number of new or previously failed permalinks "
            "to request. Use 0 to process everything."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between permalinks.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save the cache after this many requests.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for each page/interface.",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Display the browser instead of running headlessly.",
    )

    return parser.parse_args()


def normalize_permalink(permalink):
    if permalink is None:
        return None

    permalink = str(permalink).strip()

    if not permalink:
        return None

    if not permalink.startswith(("http://", "https://")):
        permalink = (
            f"https://{permalink.lstrip('/')}"
        )

    return permalink


def get_last_permalink_id(permalink):
    """
    Extract the final path component.

    Example:
      reddit.com/r/.../n5g9o4y/
      -> n5g9o4y
    """
    permalink = normalize_permalink(permalink)

    if permalink is None:
        return None

    parsed = urlparse(permalink)

    path_parts = [
        part
        for part in parsed.path.split("/")
        if part
    ]

    if not path_parts:
        return None

    return path_parts[-1]


def make_old_reddit_url(permalink):
    """
    Convert reddit.com or www.reddit.com to old.reddit.com.
    """
    permalink = normalize_permalink(permalink)

    if permalink is None:
        return None

    parsed = urlparse(permalink)

    return urlunparse(
        parsed._replace(
            netloc="old.reddit.com",
        )
    )


def is_successful_cached_value(value):
    """
    Return True only for nonempty comment text that is not a known
    Reddit interface label.
    """
    if value is None:
        return False

    normalized_value = str(value).strip()

    return (
        normalized_value != ""
        and normalized_value not in INVALID_UI_TEXTS
    )


def load_cache(cache_path):
    if not cache_path.exists():
        return {}

    with cache_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        cache = json.load(file)

    if not isinstance(cache, dict):
        raise ValueError(
            f"Invalid cache in {cache_path}: "
            "expected a JSON object."
        )

    return cache


def save_cache(cache, cache_path):
    """
    Save the cache atomically.
    """
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_suffix(
        cache_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cache,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(cache_path)

    successful_count = sum(
        is_successful_cached_value(value)
        for value in cache.values()
    )

    failed_count = len(cache) - successful_count

    print(
        f"Saved cache: {successful_count:,} successful, "
        f"{failed_count:,} failed or invalid."
    )


def collect_unique_permalinks(input_root):
    """
    Collect unique permalinks in dataset, split, row, and list order.
    """
    collected = []
    seen = set()

    for dataset_name in DATASET_NAMES:
        dataset_path = input_root / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        dataset_dict = load_from_disk(
            str(dataset_path)
        )

        for split_dataset in dataset_dict.values():
            for example in split_dataset:
                for column in (
                    "permalink_a",
                    "permalink_b",
                ):
                    values = example.get(column)

                    if values is None:
                        continue

                    for value in values:
                        permalink = normalize_permalink(value)

                        if (
                            permalink is not None
                            and permalink not in seen
                        ):
                            seen.add(permalink)
                            collected.append(permalink)

    return collected


def wait_for_reddit_challenge(
    page,
    timeout_ms,
):
    """
    Wait for Reddit's JavaScript challenge form to submit and disappear.
    """
    challenge_selector = 'input[name="js_challenge"]'

    if page.locator(challenge_selector).count() == 0:
        return

    print(
        "  Reddit JavaScript challenge detected; "
        "waiting for it to complete."
    )

    page.wait_for_selector(
        challenge_selector,
        state="detached",
        timeout=timeout_ms,
    )

    page.wait_for_load_state(
        "domcontentloaded",
        timeout=timeout_ms,
    )


def extract_new_reddit_comment(
    page,
    permalink,
    full_comment_id,
    timeout_ms,
):
    """
    Try extracting the exact comment from Reddit's current interface.
    """
    page.goto(
        permalink,
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )

    wait_for_reddit_challenge(
        page=page,
        timeout_ms=timeout_ms,
    )

    # Wait for the exact rich-text element or the exact comment
    # container with an available comment body.
    page.wait_for_function(
        """
        fullCommentId => {
            const exactContent = document.getElementById(
                `${fullCommentId}-comment-rtjson-content`
            );

            if (exactContent) {
                return true;
            }

            const selectors = [
                `shreddit-comment[thingid="${fullCommentId}"]`,
                `shreddit-comment[thing-id="${fullCommentId}"]`,
                `shreddit-comment[id="${fullCommentId}"]`
            ];

            let comment = null;

            for (const selector of selectors) {
                comment = document.querySelector(selector);

                if (comment) {
                    break;
                }
            }

            if (!comment) {
                return false;
            }

            return Boolean(
                comment.querySelector(
                    'div[slot="comment"], '
                    + '[id$="-comment-rtjson-content"]'
                )
            );
        }
        """,
        arg=full_comment_id,
        timeout=timeout_ms,
    )

    return page.evaluate(
        """
        fullCommentId => {
            const exactContent = document.getElementById(
                `${fullCommentId}-comment-rtjson-content`
            );

            if (exactContent) {
                const text = (
                    exactContent.innerText
                    || exactContent.textContent
                    || ""
                ).trim();

                if (text) {
                    return text;
                }
            }

            const selectors = [
                `shreddit-comment[thingid="${fullCommentId}"]`,
                `shreddit-comment[thing-id="${fullCommentId}"]`,
                `shreddit-comment[id="${fullCommentId}"]`
            ];

            let comment = null;

            for (const selector of selectors) {
                comment = document.querySelector(selector);

                if (comment) {
                    break;
                }
            }

            if (!comment) {
                return null;
            }

            const content = comment.querySelector(
                'div[slot="comment"], '
                + '[id$="-comment-rtjson-content"]'
            );

            if (!content) {
                return null;
            }

            const text = (
                content.innerText
                || content.textContent
                || ""
            ).trim();

            return text || null;
        }
        """,
        full_comment_id,
    )


def extract_old_reddit_comment(
    page,
    permalink,
    full_comment_id,
    timeout_ms,
):
    """
    Extract the exact comment from Old Reddit's static HTML.
    """
    old_reddit_url = make_old_reddit_url(
        permalink
    )

    if old_reddit_url is None:
        return None

    page.goto(
        old_reddit_url,
        wait_until="domcontentloaded",
        timeout=timeout_ms,
    )

    wait_for_reddit_challenge(
        page=page,
        timeout_ms=timeout_ms,
    )

    comment_selector = (
        f'div.thing.comment[data-fullname="{full_comment_id}"]'
    )

    comment = page.locator(
        comment_selector
    ).first

    comment.wait_for(
        state="attached",
        timeout=timeout_ms,
    )

    # Limit extraction to the Markdown body of the exact comment.
    body = comment.locator(
        "div.entry div.usertext-body div.md"
    ).first

    body.wait_for(
        state="attached",
        timeout=timeout_ms,
    )

    text = body.inner_text(
        timeout=timeout_ms,
    ).strip()

    return text or None


def extract_matching_comment_text(
    page,
    permalink,
    timeout_seconds,
):
    """
    Try current Reddit first, then fall back to old.reddit.com.
    """
    final_id = get_last_permalink_id(permalink)

    if final_id is None:
        return None

    full_comment_id = f"t1_{final_id}"
    timeout_ms = int(timeout_seconds * 1000)

    try:
        text = extract_new_reddit_comment(
            page=page,
            permalink=permalink,
            full_comment_id=full_comment_id,
            timeout_ms=timeout_ms,
        )

        if text is not None:
            text = str(text).strip()

            if (
                text
                and text not in INVALID_UI_TEXTS
            ):
                return text
    except PlaywrightTimeoutError:
        print(
            "  Target comment was not rendered by new Reddit; "
            "trying old.reddit.com."
        )
    except Exception as error:
        print(
            "  New Reddit extraction failed: "
            f"{type(error).__name__}: {error}. "
            "Trying old.reddit.com."
        )

    try:
        text = extract_old_reddit_comment(
            page=page,
            permalink=permalink,
            full_comment_id=full_comment_id,
            timeout_ms=timeout_ms,
        )
    except PlaywrightTimeoutError:
        # Set a VS Code breakpoint on the next line to inspect a
        # permalink that fails on both Reddit interfaces.
        return None
    except Exception as error:
        print(
            "  Old Reddit extraction failed: "
            f"{type(error).__name__}: {error}"
        )
        return None

    if text is None:
        return None

    text = str(text).strip()

    if (
        not text
        or text in INVALID_UI_TEXTS
    ):
        return None

    return text


def scrape_permalinks(
    permalinks,
    cache,
    cache_path,
    limit,
    delay,
    save_every,
    timeout,
    headed,
):
    """
    Process new URLs and retry failed or invalid cache values.

    Successful cached comments are skipped. Cached None values, empty
    strings, and known Reddit interface labels are retried.
    """
    request_count = 0
    requests_since_save = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not headed,
            args=[
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 "
                "(X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            locale="de-DE",
            viewport={
                "width": 1440,
                "height": 1000,
            },
        )

        page = context.new_page()

        try:
            for permalink in permalinks:
                if (
                    permalink in cache
                    and is_successful_cached_value(
                        cache[permalink]
                    )
                ):
                    continue

                if (
                    limit > 0
                    and request_count >= limit
                ):
                    break

                if request_count > 0 and delay > 0:
                    time.sleep(delay)

                final_id = get_last_permalink_id(
                    permalink
                )

                progress = (
                    f"{request_count + 1}/{limit}"
                    if limit > 0
                    else str(request_count + 1)
                )

                retry_label = (
                    " [retry]"
                    if permalink in cache
                    else ""
                )

                print(
                    f"[{progress}] Reading {permalink} "
                    f"(final ID={final_id!r})"
                    f"{retry_label}"
                )

                try:
                    text = extract_matching_comment_text(
                        page=page,
                        permalink=permalink,
                        timeout_seconds=timeout,
                    )
                except PlaywrightTimeoutError as error:
                    print(
                        f"  Timed out: {error}"
                    )
                    text = None
                except Exception as error:
                    print(
                        f"  Browser error: "
                        f"{type(error).__name__}: {error}"
                    )
                    text = None

                # Overwrite failed or invalid cached values.
                cache[permalink] = text

                if text is None:
                    print(
                        f"  No comment text found. "
                        f"Current page: {page.url}"
                    )
                else:
                    print(
                        f"  Found {len(text):,} characters."
                    )

                request_count += 1
                requests_since_save += 1

                if requests_since_save >= save_every:
                    save_cache(
                        cache=cache,
                        cache_path=cache_path,
                    )
                    requests_since_save = 0
        finally:
            save_cache(
                cache=cache,
                cache_path=cache_path,
            )

            context.close()
            browser.close()

    return request_count


def permalink_list_to_text(
    permalinks,
    cache,
):
    """
    Convert an ordered permalink list into <POST>-joined text.

    Returns None if:
      - the permalink list is missing or empty;
      - any permalink is invalid;
      - any permalink has not yet been attempted;
      - none of the permalinks produced valid text.

    If every permalink was attempted but only some produced text, the
    successfully recovered texts are joined in their original order.
    """
    if permalinks is None:
        return None

    try:
        if len(permalinks) == 0:
            return None
    except TypeError:
        return None

    normalized_permalinks = [
        normalize_permalink(permalink)
        for permalink in permalinks
    ]

    if any(
        permalink is None
        for permalink in normalized_permalinks
    ):
        return None

    # Avoid populating a field when some URLs in its list have never
    # been attempted.
    if any(
        permalink not in cache
        for permalink in normalized_permalinks
    ):
        return None

    found_texts = []

    for permalink in normalized_permalinks:
        text = cache.get(permalink)

        if is_successful_cached_value(text):
            found_texts.append(
                str(text).strip()
            )

    if not found_texts:
        return None

    return POST_SEPARATOR.join(found_texts)


def save_dataset_replacing(
    dataset,
    output_path,
):
    """
    Save to a temporary directory before replacing older output.
    """
    temporary_path = output_path.with_name(
        f"{output_path.name}.temporary"
    )

    backup_path = output_path.with_name(
        f"{output_path.name}.backup"
    )

    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary path already exists: {temporary_path}. "
            "Inspect and remove it manually before continuing."
        )

    if backup_path.exists():
        raise FileExistsError(
            f"Backup path already exists: {backup_path}. "
            "Inspect, restore, or remove it manually before continuing."
        )

    dataset.save_to_disk(
        str(temporary_path)
    )

    if not output_path.exists():
        temporary_path.rename(output_path)
        return

    output_path.rename(backup_path)

    try:
        temporary_path.rename(output_path)
    except Exception:
        backup_path.rename(output_path)
        raise
    else:
        shutil.rmtree(backup_path)


def translate_and_save_datasets(
    input_root,
    output_root,
    cache,
):
    """
    Add nullable text_a/text_b string columns and save each DatasetDict.
    """
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset_name in DATASET_NAMES:
        input_path = input_root / dataset_name
        output_path = output_root / dataset_name

        dataset_dict = load_from_disk(
            str(input_path)
        )

        translated_splits = {}

        for split_name, split_dataset in dataset_dict.items():
            print(
                f"Translating dataset={dataset_name!r}, "
                f"split={split_name!r}, "
                f"rows={len(split_dataset):,}"
            )

            # Explicit string features prevent an initial all-None
            # batch from being inferred as Arrow type null.
            output_features = Features(
                dict(split_dataset.features)
            )
            output_features["text_a"] = Value("string")
            output_features["text_b"] = Value("string")

            translated_split = split_dataset.map(
                lambda example: {
                    "text_a": permalink_list_to_text(
                        permalinks=example["permalink_a"],
                        cache=cache,
                    ),
                    "text_b": permalink_list_to_text(
                        permalinks=example["permalink_b"],
                        cache=cache,
                    ),
                },
                features=output_features,
                desc=(
                    f"Filling text columns for "
                    f"{dataset_name}/{split_name}"
                ),
            )

            translated_splits[split_name] = (
                translated_split
            )

        translated_dataset = DatasetDict(
            translated_splits
        )

        save_dataset_replacing(
            dataset=translated_dataset,
            output_path=output_path,
        )

        print(
            f"Saved translated dataset to {output_path}"
        )


def main():
    args = parse_args()

    if args.limit < 0:
        raise ValueError(
            "--limit must be zero or greater."
        )

    if args.delay < 0:
        raise ValueError(
            "--delay must be zero or greater."
        )

    if args.save_every <= 0:
        raise ValueError(
            "--save-every must be greater than zero."
        )

    if args.timeout <= 0:
        raise ValueError(
            "--timeout must be greater than zero."
        )

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    cache_path = Path(args.cache_file)

    cache = load_cache(cache_path)

    permalinks = collect_unique_permalinks(
        input_root=input_root,
    )

    successful_cached_count = sum(
        permalink in cache
        and is_successful_cached_value(
            cache[permalink]
        )
        for permalink in permalinks
    )

    failed_cached_count = sum(
        permalink in cache
        and not is_successful_cached_value(
            cache[permalink]
        )
        for permalink in permalinks
    )

    unattempted_count = (
        len(permalinks)
        - successful_cached_count
        - failed_cached_count
    )

    print(
        f"Found {len(permalinks):,} unique permalinks: "
        f"{successful_cached_count:,} successful cached, "
        f"{failed_cached_count:,} failed or invalid cached and "
        f"retryable, {unattempted_count:,} unattempted."
    )

    request_count = scrape_permalinks(
        permalinks=permalinks,
        cache=cache,
        cache_path=cache_path,
        limit=args.limit,
        delay=args.delay,
        save_every=args.save_every,
        timeout=args.timeout,
        headed=args.headed,
    )

    print(
        f"Made {request_count:,} browser requests."
    )

    translate_and_save_datasets(
        input_root=input_root,
        output_root=output_root,
        cache=cache,
    )

    print(
        "Finished translating and saving all datasets."
    )


if __name__ == "__main__":
    main()