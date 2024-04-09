STRING_INNER = r'(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)'
STRING = f'"{STRING_INNER}*"'
INTEGER = r"(-)?(0|[1-9][0-9]*)"
NUMBER = rf"({INTEGER})(\.[0-9]+)?([eE][+-][0-9]+)?"
BOOLEAN = r"(true|false)"
NULL = r"null"
WHITESPACE = r"[\n ]*"

type_to_regex = {
    "string": STRING,
    "integer": INTEGER,
    "number": NUMBER,
    "boolean": BOOLEAN,
    "null": NULL,
}

DATE_TIME = r"(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]{3})?(Z)?"
DATE = r"(?:\d{4})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[1-2][0-9]|3[0-1])"
TIME = r"(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\\.[0-9]+)?(Z)?"
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
DURATION = r"^P(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+S)?)?$"
EMAIL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?<!-)$"
IDN_EMAIL = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]{1,63}(\.(xn--)?[a-zA-Z0-9-]+(-[a-zA-Z0-9-]+)*\.)+[a-zA-Z]{2,}$"

HOSTNAME = r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
IPV4 = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
IPV6 = r"^(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}$"
URI = r"^[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]*$"
URI_REFERENCE = r"^[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]*|\/?[^\s]*$"
IRI_REFERENCE = r"^[^\s]*$"
IRI = r"^[^\s]*$"

PHONE_NUM = r"^\+?[0-9]{1,3}?[-. ]?(\([0-9]{1,3}\)|[0-9]{1,3})[-. ]?[0-9]{1,4}[-. ]?[0-9]{1,4}[-. ]?[0-9]{1,9}$"
URL = r"^(https?|ftp):\/\/[^\s/$.?#].[^\s]*$"
POSTAL_CODE = r"^\d{5}(-\d{4})?$"
SSN = r"^\d{3}-\d{2}-\d{4}$"

VISA_CARD_PATTERN = r"^4[0-9]{12}(?:[0-9]{3})?(?:[- ]?[0-9]{4}){0,3}$"
MASTERCARD_PATTERN = r"^5[1-5][0-9]{14}(?:[- ]?[0-9]{4}){0,3}$"
AMERICAN_EXPRESS_CARD_PATTERN = r"^3[47][0-9]{13}(?:[- ]?[0-9]{4}){0,3}$"
DINERS_CLUB_CARD_PATTERN = r"^3(?:0[0-5]|[68][0-9])[0-9]{11}(?:[- ]?[0-9]{4}){0,3}$"
DISCOVER_CARD_PATTERN = r"^6(?:011|5[0-9]{2})[0-9]{12}(?:[- ]?[0-9]{4}){0,3}$"
JCB_CARD_PATTERN = r"^(?:2131|1800|35\d{3})\d{11}(?:[- ]?[0-9]{4}){0,3}$"

INTL_PHONE_PATTERN = r"^\+[1-9]{1}[0-9]{3,14}$"


format_to_regex = {
    "uuid": UUID,
    "date-time": DATE_TIME,
    "date": DATE,
    "time": TIME,
    "duration": DURATION,
    "email": EMAIL,
    "idn-email": IDN_EMAIL,
    "hostname": HOSTNAME,
    "ipv4": IPV4,
    "ipv6": IPV6,
    "uri": URI,
    "uri-refference": URI_REFERENCE,
    "iri": IRI,
    "iri-refference": IRI_REFERENCE,
    "visa-card": VISA_CARD_PATTERN,
    "mastercard-card": MASTERCARD_PATTERN,
    "amex-card": AMERICAN_EXPRESS_CARD_PATTERN,
    "diners-club-card": DINERS_CLUB_CARD_PATTERN,
    "discover-card": DISCOVER_CARD_PATTERN,
    "jcb-card": JCB_CARD_PATTERN,
    "intl-phone": INTL_PHONE_PATTERN,
}
