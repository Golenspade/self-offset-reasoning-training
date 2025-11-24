package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gocolly/colly/v2"
)

// Judgment 代表一条"案情-判决"对
type Judgment struct {
	Case     string    `json:"case"`      // 帖子标题 + 正文
	Verdict  string    `json:"verdict"`   // 高赞回复
	Source   string    `json:"source"`    // 来源URL
	Upvotes  int       `json:"upvotes"`   // 点赞数
	Keywords []string  `json:"keywords"`  // 关键词
	CrawlAt  time.Time `json:"crawl_at"`  // 抓取时间
}

// Config 爬虫配置
type Config struct {
	TargetForums []string // 目标贴吧
	MaxPages     int      // 最大页数
	Concurrency  int      // 并发数
	OutputFile   string   // 输出文件
}

var (
	// 关键词过滤器
	keywordPatterns = []string{
		"鉴定为", "纯纯的", "有一说一", "属于是",
		"驳回上诉", "建议", "赛博", "典中典",
	}
	
	// 广告过滤正则
	adPattern = regexp.MustCompile(`(加微信|扫码|广告|推广|代理)`)
)

func main() {
	config := Config{
		TargetForums: []string{
			"weakintellect",  // 弱智吧
			"anti_pressure",  // 抗压背锅吧
			"sunxiaochuan",   // 孙笑川吧
		},
		MaxPages:    50,
		Concurrency: 10,
		OutputFile:  "../data/raw/raw_judgments.json",
	}

	log.Println("🚀 赛博裁判长 - 语料掠夺模块启动")
	log.Printf("目标贴吧: %v\n", config.TargetForums)
	log.Printf("并发数: %d\n", config.Concurrency)

	judgments := crawlJudgments(config)
	
	log.Printf("✅ 抓取完成，共获取 %d 条判例\n", len(judgments))
	
	if err := saveJudgments(judgments, config.OutputFile); err != nil {
		log.Fatalf("❌ 保存失败: %v", err)
	}
	
	log.Printf("💾 数据已保存至: %s\n", config.OutputFile)
}

func crawlJudgments(config Config) []Judgment {
	var (
		judgments []Judgment
		mu        sync.Mutex
		wg        sync.WaitGroup
	)

	// 创建收集器
	c := colly.NewCollector(
		colly.Async(true),
		colly.UserAgent("Mozilla/5.0 (compatible; CyberJudge/1.0)"),
	)

	// 限制并发
	c.Limit(&colly.LimitRule{
		DomainGlob:  "*",
		Parallelism: config.Concurrency,
		Delay:       1 * time.Second,
	})

	// 解析帖子列表
	c.OnHTML(".thread-item", func(e *colly.HTMLElement) {
		title := e.ChildText(".thread-title")
		link := e.ChildAttr(".thread-link", "href")
		
		if title != "" && link != "" {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if judgment := scrapeThread(link); judgment != nil {
					mu.Lock()
					judgments = append(judgments, *judgment)
					mu.Unlock()
					log.Printf("📝 抓取成功: %s\n", title)
				}
			}()
		}
	})

	// 错误处理
	c.OnError(func(r *colly.Response, err error) {
		log.Printf("⚠️  请求失败 [%d]: %s\n", r.StatusCode, r.Request.URL)
	})

	// 访问目标页面
	for _, forum := range config.TargetForums {
		for page := 1; page <= config.MaxPages; page++ {
			url := fmt.Sprintf("https://tieba.baidu.com/f?kw=%s&pn=%d", forum, (page-1)*50)
			c.Visit(url)
		}
	}

	c.Wait()
	wg.Wait()

	return judgments
}

func scrapeThread(url string) *Judgment {
	// TODO: 实现具体的帖子抓取逻辑
	// 1. 提取标题和正文作为 Case
	// 2. 提取高赞回复作为 Verdict
	// 3. 过滤广告和无效内容
	// 4. 提取关键词
	
	// 这里是示例实现
	return &Judgment{
		Case:     "示例案情",
		Verdict:  "鉴定为纯纯的赛博乞丐",
		Source:   url,
		Upvotes:  100,
		Keywords: extractKeywords("鉴定为纯纯的赛博乞丐"),
		CrawlAt:  time.Now(),
	}
}

func extractKeywords(text string) []string {
	var keywords []string
	for _, pattern := range keywordPatterns {
		if strings.Contains(text, pattern) {
			keywords = append(keywords, pattern)
		}
	}
	return keywords
}

func saveJudgments(judgments []Judgment, filename string) error {
	// 确保目录存在
	os.MkdirAll("../data/raw", 0755)
	
	data, err := json.MarshalIndent(judgments, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(filename, data, 0644)
}

