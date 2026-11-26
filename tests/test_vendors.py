import pandas as pd

from paidtwice.vendors import UnionFind, cluster_vendors


def test_union_find_merges_transitively():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)
    assert uf.find(4) == uf.find(5)


def _vendors(rows):
    return pd.DataFrame(rows, columns=["vendor_id", "name", "tax_id", "bank_account", "zip"])


def test_clusters_by_tax_id_bank_account_and_name_plus_zip():
    frame = _vendors(
        [
            ("V1", "Acme Supply Inc.", "11-1111111", "100", "20001"),
            ("V2", "Acme Sply", "11-1111111", "200", "20002"),  # same tax id as V1
            ("V3", "Other Name", "22-2222222", "200", "20003"),  # same bank as V2
            ("V4", "The Zed Labs, LLC", "33-3333333", "300", "20004"),
            ("V5", "ZED LABS", "44-4444444", "400", "20004"),  # same name and zip as V4
            ("V6", "ZED LABS", "55-5555555", "500", "20005"),  # same name, other zip
        ]
    )
    result = cluster_vendors(frame).set_index("vendor_id")
    assert result.loc["V1", "cluster_id"] == result.loc["V2", "cluster_id"]
    assert result.loc["V2", "cluster_id"] == result.loc["V3", "cluster_id"]
    assert result.loc["V4", "cluster_id"] == result.loc["V5", "cluster_id"]
    assert result.loc["V6", "cluster_id"] != result.loc["V4", "cluster_id"]
    assert result.loc["V1", "cluster_id"] != result.loc["V4", "cluster_id"]
    assert result.loc["V1", "cluster_size"] == 3
    assert result.loc["V6", "cluster_size"] == 1


def test_blank_keys_never_link_records():
    frame = _vendors(
        [
            ("V1", "Alpha Co", "", "", ""),
            ("V2", "Beta Co", "", "", ""),
            ("V3", "Gamma", " ", "", ""),
        ]
    )
    assert cluster_vendors(frame)["cluster_id"].nunique() == 3


def test_all_duplicate_records_join_their_vendor_and_no_vendors_merge(pipeline):
    clusters = pipeline["clusters"].set_index("vendor_id")["cluster_id"]
    truth = pipeline["truth_frame"]
    records = truth[truth["entity"] == "vendor"]
    assert (records["role"] == "duplicate_record").sum() == 120
    for true_key, group in records.groupby("group_id"):
        assert clusters[group["entity_id"]].nunique() == 1, true_key
    clusters_per_true_vendor = records.assign(cluster=clusters[records["entity_id"]].to_numpy())
    assert clusters_per_true_vendor.groupby("cluster")["group_id"].nunique().max() == 1
    assert clusters.nunique() == 3880
