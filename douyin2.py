
import requests
import pandas as pd
import os
import time
import json

# ================== 配置区 ==================
RAW_COOKIES = """UIFID_TEMP=5a9ddafc2df5b1d5b452c3de63aa171cea81dc4215a756cefc72ee80e24fb54c2edda53c2a998e16f2b2dd62a4a7b5e05bda506f41d753e847c155db3aed0f7d0f8f01241862cbd72e4119bf7bc573dd; hevc_supported=true; volume_info=%7B%22isUserMute%22%3Afalse%2C%22isMute%22%3Afalse%2C%22volume%22%3A0.5%7D; n_mh=sP7W97I1N_l65-R-3Zou4yn7lqUUyFH5OpBcRlojU8M; SelfTabRedDotControl=%5B%5D; SEARCH_RESULT_LIST_TYPE=%22single%22; SEARCH_UN_LOGIN_PV_CURR_DAY=%7B%22date%22%3A1767109028672%2C%22count%22%3A1%7D; strategyABtestKey=%221768014157.308%22; passport_csrf_token=a636277606d8d72f4f952784765866bb; passport_csrf_token_default=a636277606d8d72f4f952784765866bb; bd_ticket_guard_client_web_domain=2; ttwid=1%7CWT-MqM68oXX_Sp2NHvU98L4FOsU2nZAPYoVA0py_8KI%7C1768014158%7C5a00547412014da8a3e0f9407e85c4444727b4a6efb0a82304afbb01370919ee; d_ticket=467fc11e91501e500951474738cb890ae89b2; passport_mfa_token=CjZvJ8IjerTL9hc3SYV6T6heAKoKIpwzq9w5OtQTkHecRd%2FzLGioVAbyh6TkkLVs65NGVoP2eUQaSgo8AAAAAAAAAAAAAE%2FvDnYm0pEobXtCH%2BboTyaTW1E7LphfjQp4J%2BQAyeqy00U8LG9t2hWwaWl%2BJoKVJISMEPTAhg4Y9rHRbCACIgEDc%2FqMVg%3D%3D; passport_assist_user=Cj40bc1gfzm_k-XX9qSin-SyvjAMOk1u1pHpw838p3OllkqiOD_vK7p_vE-UMIo-bZa2wVLPhzaBIvSefr815RpKCjwAAAAAAAAAAAAAT-9SdbWjRdOpj8QQs0OeieM6KJ0eHoLNmFRitiRnR78yrGaiezzw0NLA0ytUPvuIH6wQvsGGDhiJr9ZUIAEiAQPXQTLp; passport_auth_status=071e623be19f279940bdc52e3b86b136%2C; passport_auth_status_ss=071e623be19f279940bdc52e3b86b136%2C; sid_guard=f7cac8b5e20d2ff1b74934c7285a1e3d%7C1768014218%7C5184000%7CWed%2C+11-Mar-2026+03%3A03%3A38+GMT; uid_tt=4c6a5c694f280a17f4620c4061a89cf1; uid_tt_ss=4c6a5c694f280a17f4620c4061a89cf1; sid_tt=f7cac8b5e20d2ff1b74934c7285a1e3d; sessionid=f7cac8b5e20d2ff1b74934c7285a1e3d; sessionid_ss=f7cac8b5e20d2ff1b74934c7285a1e3d; session_tlb_tag=sttt%7C8%7C98rIteINL_G3STTHKFoePf_________6lKEJM5eDAvqJXRBqmcAm9KIsgHd7PDqXqN9QentFP4U%3D; is_staff_user=false; sid_ucp_v1=1.0.0-KDIxNzAwMzIyOTIwOWRhY2FmYzE2YWNmYWM2ODk3OWRjYjIwMjEzYTcKIAjO1oDrw_UBEIqDh8sGGO8xIAwwha6P9gU4AkDxB0gEGgJscSIgZjdjYWM4YjVlMjBkMmZmMWI3NDkzNGM3Mjg1YTFlM2Q; ssid_ucp_v1=1.0.0-KDIxNzAwMzIyOTIwOWRhY2FmYzE2YWNmYWM2ODk3OWRjYjIwMjEzYTcKIAjO1oDrw_UBEIqDh8sGGO8xIAwwha6P9gU4AkDxB0gEGgJscSIgZjdjYWM4YjVlMjBkMmZmMWI3NDkzNGM3Mjg1YTFlM2Q; _bd_ticket_crypt_cookie=ee359ac65c2a20b2467505fe2537d035; __security_mc_1_s_sdk_sign_data_key_web_protect=91e0f4e8-4b66-b4d0; __security_mc_1_s_sdk_cert_key=9f862aad-4c72-8fb7; __security_mc_1_s_sdk_crypt_sdk=53ca560f-4509-8df3; __security_server_data_status=1; login_time=1768014218474; publish_badge_show_info=%220%2C0%2C0%2C1768014218971%22; DiscoverFeedExposedAd=%7B%7D; is_dash_user=1; FOLLOW_LIVE_POINT_INFO=%22MS4wLjABAAAARHCn-Kp5Ojt2Re4JpWqKW9h4qvcOr7hITTkxJgOcAGU%2F1768060800000%2F0%2F0%2F1768015442721%22; FOLLOW_NUMBER_YELLOW_POINT_INFO=%22MS4wLjABAAAARHCn-Kp5Ojt2Re4JpWqKW9h4qvcOr7hITTkxJgOcAGU%2F1768060800000%2F0%2F1768014842721%2F0%22; stream_recommend_feed_params=%22%7B%5C%22cookie_enabled%5C%22%3Atrue%2C%5C%22screen_width%5C%22%3A1707%2C%5C%22screen_height%5C%22%3A960%2C%5C%22browser_online%5C%22%3Atrue%2C%5C%22cpu_core_num%5C%22%3A20%2C%5C%22device_memory%5C%22%3A8%2C%5C%22downlink%5C%22%3A10%2C%5C%22effective_type%5C%22%3A%5C%224g%5C%22%2C%5C%22round_trip_time%5C%22%3A0%7D%22; bd_ticket_guard_client_data=eyJiZC10aWNrZXQtZ3VhcmQtdmVyc2lvbiI6MiwiYmQtdGlja2V0LWd1YXJkLWl0ZXJhdGlvbi12ZXJzaW9uIjoxLCJiZC10aWNrZXQtZ3VhcmQtcmVlLXB1YmxpYy1rZXkiOiJCQWE0SDNKUXJJU0ZkbzZuL3pWeFIyUmhncFpBS2pkLzlvb3QwR1NHS3ZJRklCd2JjcWlNSGppYnpvQjZLZjZXb1ZMNTlTVzVmUnJYN0Voa0ZOVjVmVG89IiwiYmQtdGlja2V0LWd1YXJkLXdlYi12ZXJzaW9uIjoyfQ%3D%3D; biz_trace_id=ce1f13c6; bd_ticket_guard_client_data_v2=eyJyZWVfcHVibGljX2tleSI6IkJBYTRIM0pRcklTRmRvNm4velZ4UjJSaGdwWkFLamQvOW9vdDBHU0dLdklGSUJ3YmNxaU1Iamliem9CNktmNldvVkw1OVNXNWZSclg3RWhrRk5WNWZUbz0iLCJ0c19zaWduIjoidHMuMi4yNmJhZDhlYWE5MWU0NWRmMTQxYTQ5YWU2OTdmN2ViY2M1MDNmMGJkYjZjY2EzZTE0NjI1MjA0ZjMyNTZjZDEwYzRmYmU4N2QyMzE5Y2YwNTMxODYyNGNlZGExNDkxMWNhNDA2ZGVkYmViZWRkYjJlMzBmY2U4ZDRmYTAyNTc1ZCIsInJlcV9jb250ZW50Ijoic2VjX3RzIiwicmVxX3NpZ24iOiJ2UGMxVnNsOGZnZjc2ZG5CUkZEd1ZIVXlEeEVibmtKcFVKcHVkYStoUmNjPSIsInNlY190cyI6IiMzYzlDaVhKM3R4L1FWcDd0TVRBSFhUWk45Y0dPODhiUUVUSmlxbm1MTTBqQWgwNEU3ZDdGL3hhL3ZyY2oifQ%3D%3D; download_guide=%223%2F20260110%2F0%22; odin_tt=0cee7034ae8db9d9ff374b2ab74fcf8ea7d250a4354b25aec85886f7d665f38f2360aa691de05270a9b56585971137c9dbff1fcc820ee2688972b6ed845d6f7d; my_rd=2; IsDouyinActive=true; home_can_add_dy_2_desktop=%220%22"""

SEARCH_KEYWORD = "风机噪声"
MAX_PAGES = 5
SLEEP_BETWEEN_PAGES = 8

SAVE_DIR = r"D:\cxchengxu\数据爬取"
SAVE_NAME = "风机噪声爬取结果.csv"
# ===========================================


def parse_cookies(raw):
    cookies = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            cookies[k] = v
    return cookies


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_params(keyword, offset):
    """⚠️ keyword 不要自己 quote"""
    return {
        "device_platform": "webapp",
        "aid": "6383",
        "channel": "channel_pc_web",
        "search_channel": "aweme_video_web",
        "keyword": keyword,
        "search_source": "tab_search",
        "offset": offset,
        "count": 20,
        "pc_client_type": 1,
        "is_filter_search": 1,
    }


def crawl():
    ensure_dir(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)

    cookies = parse_cookies(RAW_COOKIES)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://www.douyin.com/search/{SEARCH_KEYWORD}",
    }

    url = "https://www.douyin.com/aweme/v1/web/search/item/"

    all_rows = []
    offset = 0

    for page in range(MAX_PAGES):
        params = build_params(SEARCH_KEYWORD, offset)

        r = requests.get(
            url,
            headers=headers,
            params=params,
            cookies=cookies,
            timeout=20
        )

        print(f"[INFO] Page {page+1}, status={r.status_code}")

        if r.status_code != 200:
            print("[WARN] 请求失败，终止")
            break

        data = r.json()
        items = data.get("data", [])

        if not items:
            print("[INFO] 无更多数据")
            break

        for it in items:
            aweme = it.get("aweme_info", {})
            author = aweme.get("author", {})
            stats = aweme.get("statistics", {})

            ts = aweme.get("create_time", 0)
            pub_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""

            all_rows.append({
                "keyword": SEARCH_KEYWORD,
                "title": aweme.get("desc", ""),
                "aweme_id": aweme.get("aweme_id", ""),
                "video_url": f"https://www.douyin.com/video/{aweme.get('aweme_id', '')}",
                "author": author.get("nickname", ""),
                "followers": author.get("follower_count", 0),
                "publish_time": pub_time,
                "likes": stats.get("digg_count", 0),
                "comments": stats.get("comment_count", 0),
                "shares": stats.get("share_count", 0),
            })

        print(f"[INFO] 本页抓到 {len(items)} 条")
        offset += 20
        time.sleep(SLEEP_BETWEEN_PAGES)

    if all_rows:
        df = pd.DataFrame(all_rows)
        if os.path.exists(save_path):
            df.to_csv(save_path, mode="a", index=False, header=False, encoding="utf_8_sig")
        else:
            df.to_csv(save_path, index=False, encoding="utf_8_sig")

        print(f"[DONE] 保存 {len(all_rows)} 条 → {save_path}")
    else:
        print("[DONE] 未抓到数据")


if __name__ == "__main__":
    crawl()
