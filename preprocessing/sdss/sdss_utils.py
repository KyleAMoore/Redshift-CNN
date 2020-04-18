from hashlib import sha1


def get_guid(rows, col_name='specObjID'):
    str_cols = ''
    for idx, row in rows.iterrows():
        str_cols += str(row[col_name])
    return sha1(str_cols.encode('utf-8')).hexdigest()
