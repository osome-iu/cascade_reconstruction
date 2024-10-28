"""
A data model class for raw Twitter V1 data.
"""

import datetime

from midterm.utils import get_dict_val

TWITTER_DATE_STRING_FORMAT = "%a %b %d %H:%M:%S %z %Y"


class Tweet:
    """
    Class to handle tweet object (V1 API)
    """

    def __init__(self, post_object):
        """
        This function initializes the instance by binding the post_object

        Parameters:
            - post_object (dict): the JSON object of a tweet
        """
        self.post_object = post_object

        # Check for nested status objects
        self.is_quote = "quoted_status" in self.post_object
        if self.is_quote:
            self.quote_object = Tweet(self.post_object["quoted_status"])

        self.is_retweet = "retweeted_status" in self.post_object
        if self.is_retweet:
            self.retweet_object = Tweet(self.post_object["retweeted_status"])

        self.is_extended = "extended_tweet" in self.post_object
        if self.is_extended:
            self.extended_object = Tweet(self.post_object["extended_tweet"])

    def get_value(self, key_list: list = []):
        """
        This is the same as the midterm.get_dict_val() function
        Return `dictionary` value at the end of the key path provided
        in `key_list`.
        Indicate what value to return based on the key_list provided.
        For example, from left to right, each string in the key_list
        indicates another nested level further down in the dictionary.
        If no value is present, a `None` is returned.
        Parameters:
        ----------
        - dictionary (dict) : the dictionary object to traverse
        - key_list (list) : list of strings indicating what dict_obj
            item to retrieve
        Returns:
        ----------
        - key value (if present) or None (if not present)
        """
        return get_dict_val(self.post_object, key_list)

    def is_valid(self):
        """
        Check if the tweet object is valid.
        A valid tweet should at least have the following attributes:
            [id_str, user, text, created_at]
        """
        attributes_to_check = ["id_str", "user", "text", "created_at"]
        for attribute in attributes_to_check:
            if attribute not in self.post_object:
                return False
        return True

    def get_created_at(self):
        """
        Return tweet created_at time (str)
        """
        return self.get_value(["created_at"])

    def get_timestamp(self):
        """
        Return tweet timestamp (int)
        """
        created_at = self.get_value(["created_at"])
        timestamp = datetime.datetime.strptime(
            created_at, TWITTER_DATE_STRING_FORMAT
        ).timestamp()
        return int(timestamp)

    def get_post_ID(self):
        """
        Return the ID of the tweet (str)
        This is different from the id of the retweeted tweet or
        quoted tweet
        """
        return self.get_value(["id_str"])

    def get_user_ID(self):
        """
        Return the ID of the base-level user (str)
        """
        return self.get_value(["user", "id_str"])

    def get_user_screenname(self):
        """
        Return the screen_name of the user (str)
        """
        return self.get_value(["user", "screen_name"])

    def get_retweeted_post_ID(self):
        """
        Return the original post ID (str)
        This is retweeted tweet ID for retweet, quoted tweet ID for quote
        """
        if self.is_retweet:
            return self.retweet_object.get_post_ID()
        if self.is_quote:
            return self.quote_object.get_post_ID()
        return None

    def get_retweeted_user_ID(self):
        """
        Return the original user ID (str)
        This is retweeted user ID for retweet, quoted user ID for quote
        """
        if self.is_retweet:
            return self.retweet_object.get_user_ID()
        if self.is_quote:
            return self.quote_object.get_user_ID()
        return None

    def get_text(self):
        """
        Extract the tweet text (str)
        It will return the full_text in extended_tweet in its presence
        """

        if self.is_extended:
            text = self.extended_object.get_value(["full_text"])
        else:
            text = self.get_value(["text"])
        return text

    def get_follower_count(self):
        """
        Extract the entities from the tweet in the form of a dictionary
        """
        return self.get_value(["user", "followers_count"])

    def get_urls(self):
        """
        Get all URLs from tweet, excluding links to the tweet itself.
        All URLs are guaranteed to be in the "entities" field (no need to extract from text)
        Prioritize extraction from "extended_tweet".

        Returns:
        - url_list (list of str) : A list of URL strs
        """

        url_list = []
        if self.is_extended:
            url_objects = self.extended_object.get_value(["entities", "urls"])
        else:
            url_objects = self.get_value(["entities", "urls"])

        if url_objects is not None:
            for item in url_objects:
                expanded_url = get_dict_val(item, ["expanded_url"])
                if (expanded_url is not None) and ("twitter.com" not in expanded_url):
                    url = expanded_url
                else:
                    url = get_dict_val(item, ["url"])
                url_list.append(url)

        return url_list

    def get_hashtags(self):
        """
        Get all hashtags from the tweet, '#' symbols are excluded.
        They can be found in the "entities" field.

        Returns:
            - A list of strings representing the hashtags,
        """
        # Prioritize values from "extended_tweet" if present.
        if self.is_extended:
            raw_hashtags = self.extended_object.get_value(["entities", "hashtags"])
        else:
            raw_hashtags = self.get_value(["entities", "hashtags"])
        if raw_hashtags is not None:
            hashtags = [ht["text"] for ht in raw_hashtags]

        return hashtags

    def get_link_to_post(self):
        """
        Return the link to the tweet (str)
        so that one can click it and check the tweet in a web browser
        """
        return f"https://twitter.com/{self.get_user_screenname()}/status/{self.get_post_ID()}"

    def __repr__(self):
        """
        Define the representation of the object.
        """
        return "".join(
            [
                f"{self.__class__.__name__} object from @{self.get_user_screenname()}\n",
                f"Link: {self.get_link_to_post()}",
            ]
        )
