pub(crate) fn concurrency_limit(configured: usize, total: usize) -> usize {
    configured.max(1).min(total.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrency_limit_is_never_zero_and_never_above_total() {
        assert_eq!(concurrency_limit(0, 0), 1);
        assert_eq!(concurrency_limit(8, 3), 3);
        assert_eq!(concurrency_limit(2, 10), 2);
    }
}
