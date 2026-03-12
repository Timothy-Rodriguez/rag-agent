// main.go
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"
)

type Query struct {
	Question       string  `json:"question" binding:"required"`
	TopK           int     `json:"top_k,omitempty"`
	LanguageFilter *string `json:"language_filter,omitempty"`
}

func main() {
	r := gin.Default()
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	r.POST("/api/ask", func(c *gin.Context) {
		var q Query
		if err := c.ShouldBindJSON(&q); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		if q.TopK == 0 {
			q.TopK = 10
		}

		payload := map[string]any{
			"question": q.Question,
			"top_k":    q.TopK,
		}
		if q.LanguageFilter != nil {
			payload["language_filter"] = *q.LanguageFilter
		}

		body, _ := json.Marshal(payload)
		resp, err := http.Post("http://127.0.0.1:8001/ask", "application/json", bytes.NewBuffer(body))
		if err != nil {
			c.String(500, "event: error\ndata: RAG service down\n\n")
			return
		}
		defer resp.Body.Close()

		c.Writer.Header().Set("Content-Type", "text/event-stream")
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if line != "" {
				c.Writer.Write([]byte(line + "\n"))
				c.Writer.Flush()
			}
		}
	})

	r.Run(":8888")
}