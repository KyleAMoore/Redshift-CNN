from hashlib import sha1


def get_guid(rows, col_name='specObjID'):
    """Return a guid for given objects by calculating SHA1 hashing"""
    str_cols = ''
    for idx, row in rows.iterrows():
        str_cols += str(row[col_name])
    return sha1(str_cols.encode('utf-8')).hexdigest()
