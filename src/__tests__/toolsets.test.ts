/**
 * Tests for the Toolset Management System
 */

// ============================================================================
// TOOLSET SYSTEM (Copy for testing - matches server-direct.ts)
// ============================================================================

interface Toolset {
  name: string;
  description: string;
  enabled: boolean;
  readOnly: boolean;
  tools: ToolDefinition[];
}

interface ToolDefinition {
  name: string;
  title: string;
  description: string;
  readOnly: boolean;
  destructive: boolean;
}

class ToolsetManager {
  private toolsets = new Map<string, Toolset>();
  private registeredTools = new Set<string>();

  registerToolset(
    name: string,
    description: string,
    readOnly: boolean,
    tools: ToolDefinition[],
    enabled: boolean = true
  ): void {
    this.toolsets.set(name, {
      name,
      description,
      enabled,
      readOnly,
      tools,
    });

    for (const tool of tools) {
      this.registeredTools.add(tool.name);
    }
  }

  enableToolset(name: string): boolean {
    const toolset = this.toolsets.get(name);
    if (!toolset) return false;
    if (name === 'toolsets') return false;
    toolset.enabled = true;
    return true;
  }

  disableToolset(name: string): boolean {
    const toolset = this.toolsets.get(name);
    if (!toolset) return false;
    if (name === 'toolsets') return false;
    toolset.enabled = false;
    return true;
  }

  isToolsetEnabled(name: string): boolean {
    return this.toolsets.get(name)?.enabled ?? false;
  }

  isToolEnabled(toolName: string): boolean {
    for (const toolset of this.toolsets.values()) {
      if (toolset.tools.some(t => t.name === toolName)) {
        return toolset.enabled;
      }
    }
    return false;
  }

  getAllToolsets(): Array<{
    name: string;
    description: string;
    enabled: boolean;
    readOnly: boolean;
    toolCount: number;
    tools: string[];
  }> {
    return Array.from(this.toolsets.values()).map(ts => ({
      name: ts.name,
      description: ts.description,
      enabled: ts.enabled,
      readOnly: ts.readOnly,
      toolCount: ts.tools.length,
      tools: ts.tools.map(t => t.name),
    }));
  }

  getToolsetTools(name: string): ToolDefinition[] | null {
    const toolset = this.toolsets.get(name);
    return toolset ? toolset.tools : null;
  }

  getEnabledTools(): ToolDefinition[] {
    const tools: ToolDefinition[] = [];
    for (const toolset of this.toolsets.values()) {
      if (toolset.enabled) {
        tools.push(...toolset.tools);
      }
    }
    return tools;
  }

  getStats(): {
    totalToolsets: number;
    enabledToolsets: number;
    totalTools: number;
    enabledTools: number;
  } {
    let enabledToolsets = 0;
    let totalTools = 0;
    let enabledTools = 0;

    for (const toolset of this.toolsets.values()) {
      totalTools += toolset.tools.length;
      if (toolset.enabled) {
        enabledToolsets++;
        enabledTools += toolset.tools.length;
      }
    }

    return {
      totalToolsets: this.toolsets.size,
      enabledToolsets,
      totalTools,
      enabledTools,
    };
  }
}

// ============================================================================
// TESTS
// ============================================================================

describe('ToolsetManager', () => {
  let manager: ToolsetManager;

  beforeEach(() => {
    manager = new ToolsetManager();
  });

  describe('registerToolset', () => {
    it('should register a toolset with tools', () => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [
          { name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false },
          { name: 'push_file', title: 'Push File', description: 'Push a file', readOnly: false, destructive: true },
        ],
        true
      );

      const toolsets = manager.getAllToolsets();
      expect(toolsets).toHaveLength(1);
      expect(toolsets[0].name).toBe('repos');
      expect(toolsets[0].toolCount).toBe(2);
    });

    it('should register toolset as disabled when enabled=false', () => {
      manager.registerToolset(
        'dangerous',
        'Dangerous operations',
        false,
        [{ name: 'delete_all', title: 'Delete All', description: 'Delete everything', readOnly: false, destructive: true }],
        false
      );

      expect(manager.isToolsetEnabled('dangerous')).toBe(false);
    });

    it('should track all tool names', () => {
      manager.registerToolset(
        'test',
        'Test toolset',
        true,
        [
          { name: 'tool_a', title: 'Tool A', description: 'A', readOnly: true, destructive: false },
          { name: 'tool_b', title: 'Tool B', description: 'B', readOnly: true, destructive: false },
        ]
      );

      const toolset = manager.getAllToolsets()[0];
      expect(toolset.tools).toContain('tool_a');
      expect(toolset.tools).toContain('tool_b');
    });
  });

  describe('enableToolset', () => {
    beforeEach(() => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [{ name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false }],
        false
      );
    });

    it('should enable a disabled toolset', () => {
      expect(manager.isToolsetEnabled('repos')).toBe(false);
      const result = manager.enableToolset('repos');
      expect(result).toBe(true);
      expect(manager.isToolsetEnabled('repos')).toBe(true);
    });

    it('should return false for non-existent toolset', () => {
      const result = manager.enableToolset('nonexistent');
      expect(result).toBe(false);
    });

    it('should not allow enabling the toolsets meta-toolset', () => {
      manager.registerToolset(
        'toolsets',
        'Meta toolset',
        true,
        [{ name: 'list_toolsets', title: 'List', description: 'List toolsets', readOnly: true, destructive: false }],
        true
      );

      // Disable first (even though it won't work)
      manager.disableToolset('toolsets');
      // Try to enable
      const result = manager.enableToolset('toolsets');
      expect(result).toBe(false);
    });
  });

  describe('disableToolset', () => {
    beforeEach(() => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [{ name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false }],
        true
      );
    });

    it('should disable an enabled toolset', () => {
      expect(manager.isToolsetEnabled('repos')).toBe(true);
      const result = manager.disableToolset('repos');
      expect(result).toBe(true);
      expect(manager.isToolsetEnabled('repos')).toBe(false);
    });

    it('should return false for non-existent toolset', () => {
      const result = manager.disableToolset('nonexistent');
      expect(result).toBe(false);
    });

    it('should not allow disabling the toolsets meta-toolset', () => {
      manager.registerToolset(
        'toolsets',
        'Meta toolset',
        true,
        [{ name: 'list_toolsets', title: 'List', description: 'List toolsets', readOnly: true, destructive: false }],
        true
      );

      const result = manager.disableToolset('toolsets');
      expect(result).toBe(false);
      expect(manager.isToolsetEnabled('toolsets')).toBe(true);
    });
  });

  describe('isToolEnabled', () => {
    beforeEach(() => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [
          { name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false },
          { name: 'push_file', title: 'Push File', description: 'Push a file', readOnly: false, destructive: true },
        ],
        true
      );
      manager.registerToolset(
        'issues',
        'Issue operations',
        true,
        [{ name: 'list_issues', title: 'List Issues', description: 'List issues', readOnly: true, destructive: false }],
        false
      );
    });

    it('should return true for tools in enabled toolsets', () => {
      expect(manager.isToolEnabled('get_file')).toBe(true);
      expect(manager.isToolEnabled('push_file')).toBe(true);
    });

    it('should return false for tools in disabled toolsets', () => {
      expect(manager.isToolEnabled('list_issues')).toBe(false);
    });

    it('should return false for non-existent tools', () => {
      expect(manager.isToolEnabled('nonexistent_tool')).toBe(false);
    });
  });

  describe('getToolsetTools', () => {
    beforeEach(() => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [
          { name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false },
          { name: 'push_file', title: 'Push File', description: 'Push a file', readOnly: false, destructive: true },
        ],
        true
      );
    });

    it('should return tools for existing toolset', () => {
      const tools = manager.getToolsetTools('repos');
      expect(tools).not.toBeNull();
      expect(tools).toHaveLength(2);
      expect(tools![0].name).toBe('get_file');
      expect(tools![1].name).toBe('push_file');
    });

    it('should return null for non-existent toolset', () => {
      const tools = manager.getToolsetTools('nonexistent');
      expect(tools).toBeNull();
    });
  });

  describe('getEnabledTools', () => {
    beforeEach(() => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [
          { name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false },
        ],
        true
      );
      manager.registerToolset(
        'issues',
        'Issue operations',
        true,
        [{ name: 'list_issues', title: 'List Issues', description: 'List issues', readOnly: true, destructive: false }],
        false
      );
      manager.registerToolset(
        'search',
        'Search operations',
        true,
        [{ name: 'search_code', title: 'Search Code', description: 'Search code', readOnly: true, destructive: false }],
        true
      );
    });

    it('should return only tools from enabled toolsets', () => {
      const tools = manager.getEnabledTools();
      expect(tools).toHaveLength(2);
      expect(tools.map(t => t.name)).toContain('get_file');
      expect(tools.map(t => t.name)).toContain('search_code');
      expect(tools.map(t => t.name)).not.toContain('list_issues');
    });

    it('should update when toolsets are enabled/disabled', () => {
      manager.enableToolset('issues');
      let tools = manager.getEnabledTools();
      expect(tools).toHaveLength(3);
      expect(tools.map(t => t.name)).toContain('list_issues');

      manager.disableToolset('repos');
      tools = manager.getEnabledTools();
      expect(tools).toHaveLength(2);
      expect(tools.map(t => t.name)).not.toContain('get_file');
    });
  });

  describe('getStats', () => {
    beforeEach(() => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [
          { name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false },
          { name: 'push_file', title: 'Push File', description: 'Push a file', readOnly: false, destructive: true },
        ],
        true
      );
      manager.registerToolset(
        'issues',
        'Issue operations',
        true,
        [{ name: 'list_issues', title: 'List Issues', description: 'List issues', readOnly: true, destructive: false }],
        false
      );
    });

    it('should return correct statistics', () => {
      const stats = manager.getStats();
      expect(stats.totalToolsets).toBe(2);
      expect(stats.enabledToolsets).toBe(1);
      expect(stats.totalTools).toBe(3);
      expect(stats.enabledTools).toBe(2);
    });

    it('should update statistics when toolsets change', () => {
      manager.enableToolset('issues');
      const stats = manager.getStats();
      expect(stats.enabledToolsets).toBe(2);
      expect(stats.enabledTools).toBe(3);
    });
  });

  describe('getAllToolsets', () => {
    it('should return all toolsets with correct structure', () => {
      manager.registerToolset(
        'repos',
        'Repository operations',
        false,
        [
          { name: 'get_file', title: 'Get File', description: 'Get a file', readOnly: true, destructive: false },
        ],
        true
      );
      manager.registerToolset(
        'search',
        'Search operations',
        true,
        [{ name: 'search_code', title: 'Search Code', description: 'Search code', readOnly: true, destructive: false }],
        false
      );

      const toolsets = manager.getAllToolsets();
      expect(toolsets).toHaveLength(2);

      const repos = toolsets.find(ts => ts.name === 'repos');
      expect(repos).toBeDefined();
      expect(repos!.description).toBe('Repository operations');
      expect(repos!.enabled).toBe(true);
      expect(repos!.readOnly).toBe(false);
      expect(repos!.toolCount).toBe(1);
      expect(repos!.tools).toContain('get_file');

      const search = toolsets.find(ts => ts.name === 'search');
      expect(search).toBeDefined();
      expect(search!.enabled).toBe(false);
      expect(search!.readOnly).toBe(true);
    });
  });
});

describe('Toolset Integration', () => {
  let manager: ToolsetManager;

  beforeEach(() => {
    manager = new ToolsetManager();

    // Register toolsets like the real server
    manager.registerToolset(
      'repos',
      'Repository operations: read/write files, branches, commits',
      false,
      [
        { name: 'get_file_contents', title: 'Get File Contents', description: 'Read a file', readOnly: true, destructive: false },
        { name: 'batch_read_files', title: 'Batch Read Files', description: 'Read multiple files', readOnly: true, destructive: false },
        { name: 'get_diff', title: 'Get Diff', description: 'Diff refs', readOnly: true, destructive: false },
        { name: 'suggest_changes', title: 'Suggest Changes', description: 'Suggest changes', readOnly: true, destructive: false },
        { name: 'create_or_update_file', title: 'Create or Update File', description: 'Write a file', readOnly: false, destructive: true },
        { name: 'apply_patch', title: 'Apply Patch', description: 'Apply patch', readOnly: false, destructive: true },
        { name: 'push_files', title: 'Push Files', description: 'Commit multiple files', readOnly: false, destructive: true },
        { name: 'create_branch', title: 'Create Branch', description: 'Create branch', readOnly: false, destructive: false },
        { name: 'list_commits', title: 'List Commits', description: 'List commits', readOnly: true, destructive: false },
      ],
      true
    );

    manager.registerToolset(
      'issues',
      'Issue tracking',
      false,
      [
        { name: 'list_issues', title: 'List Issues', description: 'List issues', readOnly: true, destructive: false },
        { name: 'get_issue', title: 'Get Issue', description: 'Get issue', readOnly: true, destructive: false },
        { name: 'search_issues', title: 'Search Issues', description: 'Search issues', readOnly: true, destructive: false },
        { name: 'list_issue_comments', title: 'List Issue Comments', description: 'List comments', readOnly: true, destructive: false },
        { name: 'create_issue', title: 'Create Issue', description: 'Create issue', readOnly: false, destructive: true },
        { name: 'update_issue', title: 'Update Issue', description: 'Update issue', readOnly: false, destructive: true },
        { name: 'add_issue_comment', title: 'Add Issue Comment', description: 'Add comment', readOnly: false, destructive: true },
      ],
      true
    );

    manager.registerToolset(
      'pulls',
      'Pull request operations',
      false,
      [
        { name: 'create_pull_request', title: 'Create PR', description: 'Create PR', readOnly: false, destructive: true },
        { name: 'update_pull_request', title: 'Update PR', description: 'Update PR', readOnly: false, destructive: true },
        { name: 'get_pull_request', title: 'Get PR', description: 'Get PR details', readOnly: true, destructive: false },
        { name: 'list_pull_requests', title: 'List PRs', description: 'List PRs', readOnly: true, destructive: false },
        { name: 'get_pull_request_files', title: 'Get PR Files', description: 'Get PR files', readOnly: true, destructive: false },
        { name: 'get_pull_request_comments', title: 'Get PR Comments', description: 'Get PR comments', readOnly: true, destructive: false },
        { name: 'get_pull_request_reviews', title: 'Get PR Reviews', description: 'Get PR reviews', readOnly: true, destructive: false },
        { name: 'create_pull_request_review', title: 'Create PR Review', description: 'Create PR review', readOnly: false, destructive: true },
        { name: 'review_pr', title: 'Review PR', description: 'Review PR', readOnly: true, destructive: false },
        { name: 'merge_pull_request', title: 'Merge PR', description: 'Merge PR', readOnly: false, destructive: true },
      ],
      true
    );

    manager.registerToolset(
      'search',
      'Search tools',
      true,
      [
        { name: 'search_code', title: 'Search Code', description: 'Search code', readOnly: true, destructive: false },
        { name: 'search_repositories', title: 'Search Repos', description: 'Search repos', readOnly: true, destructive: false },
      ],
      true
    );

    manager.registerToolset(
      'utility',
      'Utility tools',
      false,
      [
        { name: 'validate_build', title: 'Validate Build', description: 'Validate build', readOnly: true, destructive: false },
        { name: 'check_github_rate_limit', title: 'Check Rate Limit', description: 'Check rate limit', readOnly: true, destructive: false },
        { name: 'clear_cache', title: 'Clear Cache', description: 'Clear cache', readOnly: false, destructive: false },
      ],
      true
    );

    manager.registerToolset(
      'toolsets',
      'Toolset management',
      false,
      [
        { name: 'list_toolsets', title: 'List Toolsets', description: 'List toolsets', readOnly: true, destructive: false },
        { name: 'get_toolset', title: 'Get Toolset', description: 'Get toolset details', readOnly: true, destructive: false },
        { name: 'enable_toolset', title: 'Enable Toolset', description: 'Enable toolset', readOnly: false, destructive: false },
        { name: 'disable_toolset', title: 'Disable Toolset', description: 'Disable toolset', readOnly: false, destructive: false },
      ],
      true
    );
  });

  it('should have all expected toolsets registered', () => {
    const toolsets = manager.getAllToolsets();
    expect(toolsets.map(ts => ts.name)).toEqual(['repos', 'issues', 'pulls', 'search', 'utility', 'toolsets']);
  });

  it('should correctly count tools across toolsets', () => {
    const stats = manager.getStats();
    expect(stats.totalToolsets).toBe(6);
    expect(stats.totalTools).toBe(35); // 9 + 7 + 10 + 2 + 3 + 4
    expect(stats.enabledToolsets).toBe(6);
    expect(stats.enabledTools).toBe(35);
  });

  it('should disable repos toolset and update counts', () => {
    manager.disableToolset('repos');
    const stats = manager.getStats();
    expect(stats.enabledToolsets).toBe(5);
    expect(stats.enabledTools).toBe(26); // 35 - 9
    expect(manager.isToolEnabled('get_file_contents')).toBe(false);
    expect(manager.isToolEnabled('list_issues')).toBe(true);
  });

  it('should not disable toolsets meta-toolset', () => {
    const result = manager.disableToolset('toolsets');
    expect(result).toBe(false);
    expect(manager.isToolsetEnabled('toolsets')).toBe(true);
    expect(manager.isToolEnabled('list_toolsets')).toBe(true);
  });

  it('should identify read-only vs read-write toolsets', () => {
    const toolsets = manager.getAllToolsets();
    const issues = toolsets.find(ts => ts.name === 'issues')!;
    const repos = toolsets.find(ts => ts.name === 'repos')!;

    expect(issues.readOnly).toBe(false);
    expect(repos.readOnly).toBe(false);
  });

  it('should identify destructive tools', () => {
    const reposTools = manager.getToolsetTools('repos')!;
    const destructiveTools = reposTools.filter(t => t.destructive);
    const nonDestructiveTools = reposTools.filter(t => !t.destructive);

    expect(destructiveTools.map(t => t.name)).toEqual(['create_or_update_file', 'apply_patch', 'push_files']);
    expect(nonDestructiveTools.map(t => t.name)).toEqual(['get_file_contents', 'batch_read_files', 'get_diff', 'suggest_changes', 'create_branch', 'list_commits']);
  });
});
