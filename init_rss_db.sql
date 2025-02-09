-- Create feeds table
CREATE TABLE IF NOT EXISTS rss_feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    category TEXT NOT NULL,
    last_updated DATETIME,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- Create articles table with unique constraint
CREATE TABLE IF NOT EXISTS rss_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    link TEXT NOT NULL,
    published_date DATETIME,
    FOREIGN KEY(feed_id) REFERENCES rss_feeds(id) ON DELETE CASCADE,
    UNIQUE(feed_id, link)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_feeds_user ON rss_feeds(user_id);
CREATE INDEX IF NOT EXISTS idx_articles_feed ON rss_articles(feed_id);
CREATE INDEX IF NOT EXISTS idx_articles_date ON rss_articles(published_date);