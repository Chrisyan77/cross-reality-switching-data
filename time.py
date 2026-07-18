import time
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

def get_chrome_driver(driver_path=None, headless=False):
    """
    获取Chrome浏览器驱动，默认为非无头模式，可以改成headless=True静默执行
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    
    if driver_path:
        driver = webdriver.Chrome(executable_path=driver_path, options=chrome_options)
    else:
        # 假设 ChromeDriver 已加到系统 PATH
        driver = webdriver.Chrome(options=chrome_options)
    return driver

def parse_news_article(driver):
    """
    在新闻详情页，提取标题和正文
    具体根据实际页面结构进行解析
    """
    try:
        # 假设标题在 <h1 class="post_title">xxx</h1> 或类似标签中
        title_elem = driver.find_element(By.CSS_SELECTOR, "h1.post_title")
        title = title_elem.text.strip()
    except NoSuchElementException:
        title = "无标题"

    # 抓取正文示例：正文可能在某个div下，如 <div class="post_content">...</div>
    try:
        content_elem = driver.find_element(By.CSS_SELECTOR, "div.post_content")
        content = content_elem.text.strip()
    except NoSuchElementException:
        content = "无正文"

    return title, content

def extract_news_date(item_element):
    """
    从列表中的每条新闻摘要处提取时间，以判断是否已经到2020年了
    页面上可能有时间标记，如 <span class="post_date">2023-08-15</span>
    根据实际结构找到相应的日期标签
    """
    try:
        date_elem = item_element.find_element(By.CSS_SELECTOR, "span.post_date")
        date_text = date_elem.text.strip()
        # 假设日期格式类似 2023-08-15
        return date_text
    except NoSuchElementException:
        return ""

def compare_date_is_older(date_str, year=2020):
    """
    将日期字符串与指定年份比较，如果date_str在year之前则返回True
    格式假设：YYYY-MM-DD
    """
    # 简单用正则或者分割判断
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if match:
        y = int(match.group(1))
        if y < year:
            return True
    return False

def main():
    driver = get_chrome_driver(headless=False)  # 如果想无头执行，可设 headless=True
    driver.get("http://vr.ithome.com")
    time.sleep(2)

    # 如果有“VR之家最新”专栏标签，需要点击它才能切换到按时间排列的列表，
    # 或者页面默认已是最新资讯。根据实际结构可能要做一次点击。例如:
    # try:
    #     tab_element = driver.find_element(By.XPATH, '//a[text()="VR之家最新"]')
    #     tab_element.click()
    #     time.sleep(2)
    # except:
    #     pass

    # 用于存储已经抓到的链接，避免重复
    visited_urls = set()

    # 准备输出文件
    if not os.path.exists("vr_news"):
        os.makedirs("vr_news")
    output_file = os.path.join("vr_news", "vr_ithome_news.txt")

    # 不断下拉或点击“加载更多”，直至出现 2020 年之前的新闻
    while True:
        time.sleep(2)
        # 找到当前页的所有新闻项目
        news_items = driver.find_elements(By.CSS_SELECTOR, "div.post_item")
        # 如果页面结构不同，可以根据实际做调整

        if not news_items:
            print("未找到新闻列表元素，请检查选择器是否正确。")
            break
        
        # 标记是否到达2020
        reached_2020 = False

        for item in news_items:
            # 提取新闻链接
            # 可能在 <a class="post_link" href="xxx"> 中
            try:
                link_elem = item.find_element(By.CSS_SELECTOR, "a.post_link")
                link_url = link_elem.get_attribute("href")
            except NoSuchElementException:
                link_url = ""

            date_str = extract_news_date(item)
            
            # 如果已经是 2020 年或者更早，就标记一下
            if compare_date_is_older(date_str, year=2020):
                reached_2020 = True
                break

            # 如果该条目没获取到link或者已经访问过则跳过
            if (not link_url) or (link_url in visited_urls):
                continue

            visited_urls.add(link_url)

            # 打开详情页面，提取标题和正文
            driver.execute_script("window.open(arguments[0]);", link_url)
            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(2)

            title, content = parse_news_article(driver)

            # 写入txt
            with open(output_file, "a", encoding="utf-8") as f:
                f.write("日期：{}\n".format(date_str))
                f.write("标题：{}\n".format(title))
                f.write("内容：{}\n\n".format(content))
                f.write("=======================================\n\n")

            # 关闭标签页，切回列表页
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        if reached_2020:
            print("已经抓取到 2020 年及之前的新闻，停止爬取。")
            break

        # 寻找并点击“加载更多”按钮，若无则结束
        try:
            load_more_btn = driver.find_element(By.XPATH, '//button[contains(text(),"加载更多")]')
            # 有时还会是 <a> 或其他标签，可根据实际修改
            load_more_btn.click()
        except NoSuchElementException:
            print("未找到'加载更多'按钮，可能已经到底或页面结构变动，停止爬取。")
            break

    driver.quit()
    print("爬取完成，结果已保存至：", output_file)

if __name__ == "__main__":
    main()